def create_model(opt):
    model = None
    if opt.model == 'two_pix2pix':
        from field_selection.models.two_pix2pix_model import TwoPix2PixModel
        model = TwoPix2PixModel()
    model.initialize(opt)
    return model
