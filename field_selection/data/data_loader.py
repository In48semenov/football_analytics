def CreateDataLoader(opt):
    from field_selection.data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader
