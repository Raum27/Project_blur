from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


def Model_pretrain():
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights).to('cuda')
    return model.eval(),weights


# 2000 = 370 second  
# 370 = 6 minuts

# 70094 = 12950
# 12950 = 215 minuts
# 215 = 4