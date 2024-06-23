from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def adversarial_loss(y_hat, y):
    return bce_loss(y_hat, y)