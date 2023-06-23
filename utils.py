import numpy as np
import random
import paddle

from paddle.metric import Accuracy


def cross_validation(model, classifier, dataloader, metric):
    classifier.eval()
    metric.reset()
    with paddle.no_grad():
        for batch_data in dataloader:
            imgs, coords, labels, idxs = batch_data
            outputs = classifier(model(coords))
            metric.update(outputs, labels)
    
    accuracy = metric.accumulate()
    return accuracy

def train(model, classifier, train_dataloader, test_dataloader, num_epochs):
    criterion = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(parameters=classifier.parameters())
    metric = Accuracy()
    # modulate = model.latent.latent_vector.detach()
    for epoch in range(num_epochs):
        classifier.train()
        for batch_data in train_dataloader:
            imgs, coords, labels, idxs = batch_data
            # print(imgs.shape,modulate.shape)
            features = model(coords)
            features=features.transpose([0,3,1,2])
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
        
        accuracy = cross_validation(model, classifier, test_dataloader, metric)
        print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.4f}")




def get_coordinate_grid(res: int, centered: bool = True):
    """Returns a normalized coordinate grid for a res by res sized image.
      Args:
        res (int): Resolution of image.
        centered (bool): If True assumes coordinates lie at pixel centers. This is
          equivalent to the align_corners argument in Pytorch. This should always be
          set to True as this ensures we are consistent across different
          resolutions, but keep False as option for backwards compatibility.

      Returns:
        Jnp array of shape (height, width, 2).

      Notes:
        Output will be in [0, 1] (i.e. coordinates are normalized to lie in [0, 1]).
    """
    if centered:
        half_pixel = 1. / (2. * res)  # Size of half a pixel in grid
        coords_one_dim = np.linspace(half_pixel, 1. - half_pixel, res)
    else:
        coords_one_dim = np.linspace(0, 1, res)
    # Array will have shape (height, width, 2)
    return np.stack(
        np.meshgrid(coords_one_dim, coords_one_dim, indexing='ij'), axis=-1)


def worker_init_fn(worker_id):
    """
    dataloader worker的初始函数
    根据worker_id设置不同的随机数种子，避免多个worker输出相同的数据
    """
    np.random.seed(random.randint(0, 100000))


def get_strategy():
    """
    TBD
    """
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.fuse_elewise_add_act_ops = True
    build_strategy.fuse_bn_act_ops = True
    build_strategy.enable_auto_fusion = True
    build_strategy.fuse_relu_depthwise_conv = True
    build_strategy.fuse_broadcast_ops = True
    build_strategy.fuse_all_optimizer_ops = True
    build_strategy.enable_inplace = True

    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.build_strategy = build_strategy
    return strategy


