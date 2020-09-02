from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import hyperparams as hp


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step ** 0.5 * min(step_num * warmup_step ** -1.5, step_num ** -0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 画loss图
def plot_loss_metrics(history_loss, global_step, title):
    fig = plt.figure()
    plt.plot(global_step, history_loss,
             color='b', label=title)

    plt.legend()
    fig.savefig(os.path.join(hp.loss_path, title), dpi=200)
    plt.close('all')


def main():
    dataset = get_dataset()
    global_step = 0

    m = nn.DataParallel(Model().cuda())

    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr,weight_decay=1e-6)

    pos_weight = t.FloatTensor([5.]).cuda()
    writer = SummaryWriter()

    mel_loss_history = []
    post_mel_loss_history = []
    total_loss_history = []
    global_step_history = []

    for epoch in range(hp.epochs):
        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer,
                                drop_last=True, num_workers=8)

        pbar = tqdm(dataloader)

        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d" % epoch)
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)

            character, mel, mel_input, pos_text, pos_mel, _ = data

            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)

            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()

            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel)


            mel_loss = nn.MSELoss()(mel_pred, mel)
            post_mel_loss = nn.MSELoss()(postnet_pred, mel)

            loss = mel_loss + post_mel_loss

            if global_step % 100 == 0:
                print('mel_loss==', mel_loss.item(), 'post_mel_loss==', post_mel_loss.item())
                print('total_loss==', loss.item())
                mel_loss_history.append(mel_loss.item())
                post_mel_loss_history.append(post_mel_loss.item())
                total_loss_history.append(loss.item())
                global_step_history.append(global_step)

                plot_loss_metrics(mel_loss_history, global_step_history, 'mel_loss.png')
                plot_loss_metrics(post_mel_loss_history, global_step_history, 'post_mel_loss.png')
                plot_loss_metrics(total_loss_history, global_step_history, 'total_loss.png')

            writer.add_scalars('training_loss', {
                'mel_loss': mel_loss,
                'post_mel_loss': post_mel_loss,

            }, global_step)

            writer.add_scalars('alphas', {
                'encoder_alpha': m.module.encoder.alpha.data,
                'decoder_alpha': m.module.decoder.alpha.data,
            }, global_step)

            if global_step % hp.image_step == 1:

                for i, prob in enumerate(attn_probs):

                    num_h = prob.size(0)
                    for j in range(4):
                        x = vutils.make_grid(prob[j * 8] * 255)
                        writer.add_image('Attention_%d_0' % global_step, x, i * 4 + j)

                for i, prob in enumerate(attns_enc):
                    num_h = prob.size(0)

                    for j in range(4):
                        x = vutils.make_grid(prob[j * 8] * 255)
                        writer.add_image('Attention_enc_%d_0' % global_step, x, i * 4 + j)

                for i, prob in enumerate(attns_dec):

                    num_h = prob.size(0)
                    for j in range(4):
                        x = vutils.make_grid(prob[j * 8] * 255)
                        writer.add_image('Attention_dec_%d_0' % global_step, x, i * 4 + j)

            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()

            nn.utils.clip_grad_norm_(m.parameters(), 1.)

            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0:
                t.save({'model': m.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(hp.checkpoint_path, 'checkpoint_transformer_%d.pth.tar' % global_step))


if __name__ == '__main__':
    main()