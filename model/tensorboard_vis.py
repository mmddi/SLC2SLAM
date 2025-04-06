from torch.utils.tensorboard import SummaryWriter
import os
import yaml

class TFVisualizer(object):
    def __init__(self, log_dir, vis_interval, config):
        self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir))
        self.vis_interval = vis_interval
        self.config = config

        # dump args to tensorboard
        args_str = '{}'.format(yaml.dump(config, sort_keys=False, indent=4))
        self.tb_writer.add_text('Exp_args', args_str, 0)

    def vis_scalars(self, i_iter, losses, names):#i_iter表示当前迭代次数，losses表示损失值列表，names表示损失值对应的名称列表。
        for i, loss in enumerate(losses):
            self.tb_writer.add_scalar(names[i], loss, i_iter)

            
    def vis_histogram(self, i_iter, value, names):
            self.tb_writer.add_histogram(tag=names, values=value, global_step=i_iter)
