import pickle
from PIL import Image
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    def __init__(self, seg_paths, bscan_paths, attributes, pair_indices, seg_transform=None,
                 bscan_transform=None):
        """
        Initilize paired dataset
        seg_paths, bscan_paths and attributes are either list or str. When they are str, they
        are path to the saved list
        """
        if isinstance(seg_paths, str):
            with open(seg_paths, 'rb'):
                seg_paths = pickle.load(seg_paths)
        self.seg_paths = seg_paths
        self.bscan_paths = bscan_paths
        if isinstance(attributes, str):
            with open(attributes, 'rb') as f:
                attributes = pickle.load(f)
        self.attributes = attributes
        self.pair_indices = pair_indices
        self.seg_transform = seg_transform
        self.bscan_transform = bscan_transform
        
    def __len__(self):
        return len(self.pair_indices)
    
    def __getitem__(self, index):
        pair = self.pair_indices[index]
        src_idx, dst_idx = pair
        src_seg_path = self.seg_paths[src_idx]
        src_bscan_path = self.bscan_paths[src_idx]
        src_attr = self.attributes[src_idx]
        dst_seg_path = self.seg_paths[dst_idx]
        dst_bscan_path = self.bscan_paths[dst_idx]
        dst_attr = self.attributes[dst_idx]
        src_seg = Image.open(src_seg_path).convert('L')
        src_bscan = Image.open(src_bscan_path).convert('L')
        dst_seg = Image.open(dst_seg_path).convert('L')
        dst_bscan = Image.open(dst_bscan_path).convert('L')
        if self.seg_transform:
            src_seg = self.seg_transform(src_seg)
            dst_seg = self.seg_transform(dst_seg)
        if self.bscan_transform:
            src_bscan = self.bscan_transform(src_bscan)
            dst_bscan = self.bscan_transform(dst_bscan)
        return src_seg, src_bscan, src_attr, dst_seg, dst_bscan, dst_attr