import torch
import torch.nn as nn


MODEL_TYPES = ("baseline", "pointnet_transformer", "edgeconv_anchor", "mmchain_lite", "mmchain_lite_st")


class PointNetEncoder(nn.Module):
    def __init__(self, input_channels=3, emb_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, emb_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return torch.max(x, 2)[0]


class PointNetTransformerPoseNet(nn.Module):
    def __init__(self, input_channels=3, num_joints=13, seq_len=10, emb_dim=256):
        super().__init__()
        self.num_joints = num_joints
        self.emb_dim = emb_dim
        self.spatial_encoder = PointNetEncoder(input_channels=input_channels, emb_dim=self.emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, self.emb_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.regressor = nn.Sequential(
            nn.Linear(self.emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_joints * 3),
        )

    def forward(self, x):
        batch_size, seq_len, channels, point_count = x.shape
        x_flat = x.reshape(batch_size * seq_len, channels, point_count)
        feats = self.spatial_encoder(x_flat)
        feats = feats.reshape(batch_size, seq_len, -1) + self.pos_embedding[:, :seq_len, :]
        temp_feats = self.transformer(feats)
        return self.regressor(temp_feats).view(batch_size, seq_len, self.num_joints, 3)


class EdgeConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim * 2, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

    def forward(self, features, knn_idx):
        neighbors = gather_neighbors(features, knn_idx)
        center = features.unsqueeze(2).expand_as(neighbors)
        edge_features = torch.cat([center, neighbors - center], dim=-1)
        edge_features = edge_features.permute(0, 3, 1, 2).contiguous()
        edge_features = self.net(edge_features)
        return torch.max(edge_features, dim=-1)[0].transpose(1, 2).contiguous()


def gather_neighbors(features, knn_idx):
    batch_size, point_count, _channels = features.shape
    neighbor_count = knn_idx.shape[-1]
    batch_idx = torch.arange(batch_size, device=features.device).view(batch_size, 1, 1)
    batch_idx = batch_idx.expand(-1, point_count, neighbor_count)
    return features[batch_idx, knn_idx]


def knn_indices(coords, k):
    point_count = coords.shape[1]
    if point_count <= 1:
        return torch.zeros(coords.shape[0], point_count, 1, dtype=torch.long, device=coords.device)

    neighbor_count = min(k + 1, point_count)
    with torch.no_grad():
        distances = torch.cdist(coords, coords)
        indices = torch.topk(distances, k=neighbor_count, largest=False, dim=-1).indices
    if neighbor_count > 1:
        indices = indices[:, :, 1:]
    return indices.contiguous()


class EdgeConvSpatialEncoder(nn.Module):
    def __init__(self, input_channels=3, emb_dim=128, k=8):
        super().__init__()
        self.k = k
        self.edge1 = EdgeConvBlock(input_channels, 64)
        self.edge2 = EdgeConvBlock(64, emb_dim)
        self.out_norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        batch_size, seq_len, channels, point_count = x.shape
        x_flat = x.reshape(batch_size * seq_len, channels, point_count)
        point_features = x_flat.transpose(1, 2).contiguous()
        coords = point_features[:, :, :3].contiguous()
        knn_idx = knn_indices(coords, self.k)

        point_features = self.edge1(point_features, knn_idx)
        point_features = self.edge2(point_features, knn_idx)
        return self.out_norm(point_features)


class LatentAnchorAggregator(nn.Module):
    def __init__(self, emb_dim=128, num_anchors=8, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchor_queries = nn.Parameter(torch.randn(num_anchors, emb_dim) * 0.02)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, point_features):
        batch_frames = point_features.shape[0]
        queries = self.anchor_queries.unsqueeze(0).expand(batch_frames, -1, -1)
        attended, _weights = self.attn(queries, point_features, point_features, need_weights=False)
        tokens = self.norm1(queries + attended)
        return self.norm2(tokens + self.ffn(tokens))


def make_anchor_positions(num_anchors):
    base = torch.tensor(
        [
            [-0.45, 0.00, 0.10],
            [0.45, 0.00, 0.10],
            [-0.50, 0.00, 0.55],
            [0.50, 0.00, 0.55],
            [-0.45, 0.00, 1.05],
            [0.45, 0.00, 1.05],
            [0.00, -0.40, 0.70],
            [0.00, 0.40, 0.70],
            [0.00, -0.35, 1.25],
            [0.00, 0.35, 1.25],
            [0.00, 0.00, 1.60],
            [0.00, 0.00, 0.95],
        ],
        dtype=torch.float32,
    )
    if num_anchors <= base.shape[0]:
        return base[:num_anchors].contiguous()

    z_count = max(1, (num_anchors + 3) // 4)
    xs = torch.tensor([-0.45, 0.45], dtype=torch.float32)
    ys = torch.tensor([-0.35, 0.35], dtype=torch.float32)
    zs = torch.linspace(0.10, 1.60, z_count, dtype=torch.float32)
    anchors = []
    for z in zs:
        for x in xs:
            for y in ys:
                anchors.append(torch.stack([x, y, z]))
    return torch.stack(anchors, dim=0)[:num_anchors].contiguous()


def anchor_knn_indices(anchor_positions, k):
    point_count = anchor_positions.shape[0]
    if point_count <= 1:
        return torch.zeros(point_count, 1, dtype=torch.long)
    neighbor_count = min(k + 1, point_count)
    distances = torch.cdist(anchor_positions.unsqueeze(0), anchor_positions.unsqueeze(0))[0]
    indices = torch.topk(distances, k=neighbor_count, largest=False, dim=-1).indices
    if neighbor_count > 1:
        indices = indices[:, 1:]
    return indices.contiguous()


class FixedAnchorAggregator(nn.Module):
    def __init__(self, emb_dim=128, num_anchors=12, num_heads=4, dropout=0.1, anchor_sigma=0.45):
        super().__init__()
        anchor_positions = make_anchor_positions(num_anchors)
        self.num_anchors = num_anchors
        self.anchor_sigma = float(anchor_sigma)
        self.register_buffer("anchor_positions", anchor_positions)
        self.anchor_queries = nn.Parameter(torch.randn(num_anchors, emb_dim) * 0.02)
        self.anchor_pos_mlp = nn.Sequential(
            nn.Linear(3, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.local_proj = nn.Sequential(
            nn.Linear(emb_dim + 3, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, point_features, coords):
        batch_frames = point_features.shape[0]
        anchors = self.anchor_positions.to(device=coords.device, dtype=coords.dtype)
        anchors_batched = anchors.unsqueeze(0).expand(batch_frames, -1, -1)

        distances = torch.cdist(anchors_batched, coords)
        weights = torch.softmax(-(distances ** 2) / max(self.anchor_sigma ** 2, 1e-6), dim=-1)
        local_features = torch.bmm(weights, point_features)
        local_offsets = torch.bmm(weights, coords) - anchors_batched

        queries = self.anchor_pos_mlp(anchors).unsqueeze(0) + self.anchor_queries.unsqueeze(0)
        queries = queries.expand(batch_frames, -1, -1)
        queries = queries + self.local_proj(torch.cat([local_features, local_offsets], dim=-1))

        attended, _weights = self.attn(queries, point_features, point_features, need_weights=False)
        tokens = self.norm1(queries + attended)
        return self.norm2(tokens + self.ffn(tokens))


class AnchorGeometryMixer(nn.Module):
    def __init__(self, emb_dim=128, num_anchors=12, k=3, dropout=0.1):
        super().__init__()
        anchor_positions = make_anchor_positions(num_anchors)
        self.register_buffer("knn_idx", anchor_knn_indices(anchor_positions, k))
        self.edge = EdgeConvBlock(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, tokens):
        batch_size = tokens.shape[0]
        knn_idx = self.knn_idx.to(tokens.device).unsqueeze(0).expand(batch_size, -1, -1)
        mixed = self.edge(tokens, knn_idx)
        return self.norm(tokens + self.dropout(mixed))


class AnchorSelfGeometryBlock(nn.Module):
    def __init__(self, emb_dim=128, num_anchors=12, num_heads=4, dropout=0.1):
        super().__init__()
        anchor_positions = make_anchor_positions(num_anchors)
        self.register_buffer("anchor_positions", anchor_positions)
        self.pos_proj = nn.Linear(3, emb_dim)
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.geometry_mixer = AnchorGeometryMixer(emb_dim, num_anchors, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, tokens):
        anchors = self.anchor_positions.to(device=tokens.device, dtype=tokens.dtype)
        pos = self.pos_proj(anchors).unsqueeze(0)
        attended, _weights = self.self_attn(tokens + pos, tokens + pos, tokens, need_weights=False)
        tokens = self.norm1(tokens + self.dropout(attended))
        tokens = self.geometry_mixer(tokens)
        return self.norm2(tokens + self.ffn(tokens))


class GeometryAwareChainBlock(nn.Module):
    def __init__(self, emb_dim=128, num_anchors=12, num_heads=4, dropout=0.1):
        super().__init__()
        anchor_positions = make_anchor_positions(num_anchors)
        self.register_buffer("anchor_positions", anchor_positions)
        self.pos_proj = nn.Linear(3, emb_dim)
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.cross_geometry = nn.Sequential(
            nn.Linear(emb_dim * 2 + 3, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.geometry_mixer = AnchorGeometryMixer(emb_dim, num_anchors, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.norm3 = nn.LayerNorm(emb_dim)

    def forward(self, current, previous):
        anchors = self.anchor_positions.to(device=current.device, dtype=current.dtype)
        pos = self.pos_proj(anchors).unsqueeze(0)
        attended, _weights = self.cross_attn(current + pos, previous + pos, previous, need_weights=False)
        current = self.norm1(current + self.dropout(attended))

        anchor_offsets = anchors.unsqueeze(0).expand(current.shape[0], -1, -1)
        geom_input = torch.cat([current, previous, anchor_offsets], dim=-1)
        current = self.norm2(current + self.cross_geometry(geom_input))
        current = self.geometry_mixer(current)
        return self.norm3(current + self.ffn(current))


class RadarChainPoseNetLite(nn.Module):
    def __init__(
        self,
        input_channels=3,
        num_joints=13,
        seq_len=10,
        emb_dim=128,
        num_anchors=8,
        edge_k=8,
        num_heads=4,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.num_anchors = num_anchors
        self.emb_dim = emb_dim

        self.spatial_encoder = EdgeConvSpatialEncoder(
            input_channels=input_channels,
            emb_dim=emb_dim,
            k=edge_k,
        )
        self.anchor_aggregator = LatentAnchorAggregator(
            emb_dim=emb_dim,
            num_anchors=num_anchors,
            num_heads=num_heads,
        )
        self.time_embedding = nn.Parameter(torch.randn(1, seq_len, 1, emb_dim) * 0.02)
        self.anchor_embedding = nn.Parameter(torch.randn(1, 1, num_anchors, emb_dim) * 0.02)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.spatial_mixer = nn.TransformerEncoder(spatial_layer, num_layers=1)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.temporal_mixer = nn.TransformerEncoder(temporal_layer, num_layers=2)

        self.pose_head = nn.Sequential(
            nn.LayerNorm(num_anchors * emb_dim),
            nn.Linear(num_anchors * emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_joints * 3),
        )

    def forward(self, x):
        batch_size, seq_len, _channels, _point_count = x.shape
        point_features = self.spatial_encoder(x)
        anchor_tokens = self.anchor_aggregator(point_features)
        anchor_tokens = anchor_tokens.view(batch_size, seq_len, self.num_anchors, self.emb_dim)
        anchor_tokens = anchor_tokens + self.time_embedding[:, :seq_len] + self.anchor_embedding

        spatial_tokens = anchor_tokens.reshape(batch_size * seq_len, self.num_anchors, self.emb_dim)
        spatial_tokens = self.spatial_mixer(spatial_tokens)
        anchor_tokens = spatial_tokens.view(batch_size, seq_len, self.num_anchors, self.emb_dim)

        temporal_tokens = anchor_tokens.permute(0, 2, 1, 3).contiguous()
        temporal_tokens = temporal_tokens.view(batch_size * self.num_anchors, seq_len, self.emb_dim)
        temporal_tokens = self.temporal_mixer(temporal_tokens)
        anchor_tokens = temporal_tokens.view(batch_size, self.num_anchors, seq_len, self.emb_dim)
        anchor_tokens = anchor_tokens.permute(0, 2, 1, 3).contiguous()

        pose_features = anchor_tokens.reshape(batch_size * seq_len, self.num_anchors * self.emb_dim)
        pose = self.pose_head(pose_features)
        return pose.view(batch_size, seq_len, self.num_joints, 3)


class MmChainPoseLite(nn.Module):
    def __init__(
        self,
        input_channels=5,
        num_joints=13,
        seq_len=5,
        emb_dim=128,
        num_anchors=12,
        edge_k=8,
        num_heads=4,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.num_anchors = num_anchors
        self.emb_dim = emb_dim

        self.spatial_encoder = EdgeConvSpatialEncoder(
            input_channels=input_channels,
            emb_dim=emb_dim,
            k=edge_k,
        )
        self.anchor_aggregator = FixedAnchorAggregator(
            emb_dim=emb_dim,
            num_anchors=num_anchors,
            num_heads=num_heads,
        )
        self.time_embedding = nn.Parameter(torch.randn(1, seq_len, 1, emb_dim) * 0.02)
        self.anchor_embedding = nn.Parameter(torch.randn(1, 1, num_anchors, emb_dim) * 0.02)
        self.self_geometry = AnchorSelfGeometryBlock(
            emb_dim=emb_dim,
            num_anchors=num_anchors,
            num_heads=num_heads,
        )
        self.chain_block = GeometryAwareChainBlock(
            emb_dim=emb_dim,
            num_anchors=num_anchors,
            num_heads=num_heads,
        )
        self.pose_head = nn.Sequential(
            nn.LayerNorm(num_anchors * emb_dim),
            nn.Linear(num_anchors * emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_joints * 3),
        )

    def forward(self, x):
        batch_size, seq_len, channels, point_count = x.shape
        point_features = self.spatial_encoder(x)
        coords = x.reshape(batch_size * seq_len, channels, point_count)
        coords = coords.transpose(1, 2).contiguous()[:, :, :3]

        anchor_tokens = self.anchor_aggregator(point_features, coords)
        anchor_tokens = anchor_tokens.view(batch_size, seq_len, self.num_anchors, self.emb_dim)
        anchor_tokens = anchor_tokens + self.time_embedding[:, :seq_len] + self.anchor_embedding

        chain_outputs = []
        previous = None
        for time_idx in range(seq_len):
            current = self.self_geometry(anchor_tokens[:, time_idx])
            if previous is not None:
                current = self.chain_block(current, previous)
            chain_outputs.append(current)
            previous = current

        chain_tokens = torch.stack(chain_outputs, dim=1)
        pose_features = chain_tokens.reshape(batch_size * seq_len, self.num_anchors * self.emb_dim)
        pose = self.pose_head(pose_features)
        return pose.view(batch_size, seq_len, self.num_joints, 3)


class MmChainPoseSpatioTemporalLite(nn.Module):
    def __init__(
        self,
        input_channels=5,
        num_joints=13,
        seq_len=5,
        emb_dim=128,
        num_anchors=12,
        edge_k=8,
        num_heads=4,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.num_anchors = num_anchors
        self.emb_dim = emb_dim

        self.spatial_encoder = EdgeConvSpatialEncoder(
            input_channels=input_channels,
            emb_dim=emb_dim,
            k=edge_k,
        )
        self.anchor_aggregator = FixedAnchorAggregator(
            emb_dim=emb_dim,
            num_anchors=num_anchors,
            num_heads=num_heads,
        )
        self.time_embedding = nn.Parameter(torch.randn(1, seq_len, 1, emb_dim) * 0.02)
        self.anchor_embedding = nn.Parameter(torch.randn(1, 1, num_anchors, emb_dim) * 0.02)
        self.self_geometry = AnchorSelfGeometryBlock(
            emb_dim=emb_dim,
            num_anchors=num_anchors,
            num_heads=num_heads,
        )

        st_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.spatio_temporal_mixer = nn.TransformerEncoder(st_layer, num_layers=2)
        self.st_norm = nn.LayerNorm(emb_dim)

        self.pose_head = nn.Sequential(
            nn.LayerNorm(num_anchors * emb_dim),
            nn.Linear(num_anchors * emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_joints * 3),
        )

    def forward(self, x):
        batch_size, seq_len, channels, point_count = x.shape
        point_features = self.spatial_encoder(x)
        coords = x.reshape(batch_size * seq_len, channels, point_count)
        coords = coords.transpose(1, 2).contiguous()[:, :, :3]

        anchor_tokens = self.anchor_aggregator(point_features, coords)
        anchor_tokens = anchor_tokens.view(batch_size, seq_len, self.num_anchors, self.emb_dim)
        anchor_tokens = anchor_tokens + self.time_embedding[:, :seq_len] + self.anchor_embedding

        frame_tokens = anchor_tokens.reshape(batch_size * seq_len, self.num_anchors, self.emb_dim)
        frame_tokens = self.self_geometry(frame_tokens)
        anchor_tokens = frame_tokens.view(batch_size, seq_len, self.num_anchors, self.emb_dim)

        # Flatten time and anchor positions so attention can directly mix all spatio-temporal tokens.
        st_tokens = anchor_tokens.reshape(batch_size, seq_len * self.num_anchors, self.emb_dim)
        st_tokens = self.spatio_temporal_mixer(st_tokens)
        st_tokens = self.st_norm(st_tokens)
        st_tokens = st_tokens.view(batch_size, seq_len, self.num_anchors, self.emb_dim)

        pose_features = st_tokens.reshape(batch_size * seq_len, self.num_anchors * self.emb_dim)
        pose = self.pose_head(pose_features)
        return pose.view(batch_size, seq_len, self.num_joints, 3)


def build_model(model_type="baseline", input_channels=3, num_joints=13, seq_len=10):
    if model_type == "baseline":
        model_type = "pointnet_transformer"

    if model_type == "pointnet_transformer":
        return PointNetTransformerPoseNet(
            input_channels=input_channels,
            num_joints=num_joints,
            seq_len=seq_len,
        )

    if model_type == "edgeconv_anchor":
        return RadarChainPoseNetLite(
            input_channels=input_channels,
            num_joints=num_joints,
            seq_len=seq_len,
        )

    if model_type == "mmchain_lite":
        return MmChainPoseLite(
            input_channels=input_channels,
            num_joints=num_joints,
            seq_len=seq_len,
        )

    if model_type == "mmchain_lite_st":
        return MmChainPoseSpatioTemporalLite(
            input_channels=input_channels,
            num_joints=num_joints,
            seq_len=seq_len,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


RadarPoseNet = PointNetTransformerPoseNet
