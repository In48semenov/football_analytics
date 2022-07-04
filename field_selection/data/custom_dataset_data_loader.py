import torch.utils.data


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'unaligned':
        from field_selection.data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()

    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader:
    def __init__(self):
        pass

    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
