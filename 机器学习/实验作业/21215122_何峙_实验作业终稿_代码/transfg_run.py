import torch
from torch import cos_, nn, optim
import torch.nn.functional as F

from urllib.request import urlretrieve
import time
import numpy as np
import matplotlib.pyplot as plt

from models.modeling import VisionTransformer, CONFIGS, Block, LayerNorm
from data_load import *
import configs


def fetch_part_attention(vit_features, att_mat_list):
    # att_mat_list为 12(层) x [batch_size, 12(head), 197, 197]
    # print(f'feature size: {vit_features.size()}')
    # print(f'att mat size: {att_mat_list[0].size()}')
    
    joint_att_mat = att_mat_list[0]   # [batch_size, 12(head_size), 197, 197]
    for n in range(1, len(att_mat_list)):
        joint_att_mat = torch.matmul(att_mat_list[n], joint_att_mat)

    # print(f'1. joint_att_mat size: {joint_att_mat.size()}')
    joint_att_mat = joint_att_mat[:, :, 0, 1:]  # [batch_size, head_size, 196]，找CLS_token attent到其他token的注意力
    # print(f'2. joint_att_mat size: {joint_att_mat.size()}')
    return joint_att_mat.max(2)[1]

def fetch_part_features(vit_features, att_index):
    att_index = att_index + 1  # CLS_token的index为0，所以要+1以从图像token里选取
    parts = []
    batch_num, head_num = att_index.shape
    for b in range(batch_num):
        parts.append(vit_features[b, att_index[b, :], :])
    parts = torch.stack(parts).squeeze(1)
    cls_feature = vit_features[:, 0, :].unsqueeze(1)
    feature_selected = torch.cat((cls_feature, parts), dim=1)
    return feature_selected


def contrastive_loss(features, targets):
    batch_num, _ = features.shape
    features = F.normalize(features)
    cos_mat = features.mm(features.t())
    pos_label_mat = targets.mm(targets.t())  # targets为one-hot矩阵
    neg_label_mat = 1 - pos_label_mat
    pos_cos_mat = 1 - cos_mat
    neg_cos_mat = cos_mat - 0.4
    neg_cos_mat[neg_cos_mat < 0] = 0
    loss = (pos_cos_mat * pos_label_mat).sum() + (neg_cos_mat * neg_label_mat).sum()
    loss /= (batch_num * batch_num)
    return loss


class PartLayer(nn.Module):

    def __init__(self, vit_config):
        super(PartLayer, self).__init__()
        self.part_transformer = Block(vit_config, vis=True)
        self.part_norm = LayerNorm(vit_config.hidden_size, eps=1e-6)

    def forward(self, vit_features, att_weight_list):
        att_part_index = fetch_part_attention(vit_features, att_weight_list)
        part_feature = fetch_part_features(vit_features, att_part_index)

        part_states, part_attention_weights = self.part_transformer(part_feature)
        part_states = self.part_norm(part_states)

        return part_states, part_attention_weights


def transfg_fine_tune(encoder, 
                      partlayer, 
                      classifier, 
                      train_dataloader, 
                      val_dataloader, 
                      test_dataloader, 
                      n_epochs, 
                      lr, 
                      device):

    encoder.to(device)
    partlayer.to(device)
    classifier.to(device)

    optimizer_encoder = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_partlayer = optim.Adam(partlayer.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr, betas=(0.5, 0.9))
    cross_criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    train_losses, val_losses, test_acc_list = [], [], []
    for epoch in range(n_epochs):
        encoder.train()
        partlayer.train()
        classifier.train()

        train_loss = 0.
        for batch_i, (batch_img_x, batch_target) in enumerate(train_dataloader):
            batch_img_x, batch_target = batch_img_x.to(device), batch_target.to(device)
            features, att_weight_list = encoder(batch_img_x)
            part_features, part_att_weights = partlayer(features, att_weight_list)

            cls_features = part_features[:, 0, :]
            logits = classifier(cls_features)

            optimizer_encoder.zero_grad()
            optimizer_partlayer.zero_grad()
            optimizer_classifier.zero_grad()
            loss = cross_criterion(logits.squeeze(-1), batch_target) + contrastive_loss(cls_features, batch_target)
            loss.backward()
            optimizer_encoder.step()
            optimizer_partlayer.step()
            optimizer_classifier.step()

            train_loss += loss.item()

        train_losses.append(train_loss)

        encoder.eval()
        partlayer.eval()
        classifier.eval()
        val_loss = 0.
        for batch_i, (batch_img, batch_tgt) in enumerate(val_dataloader):
            batch_img, batch_tgt = batch_img.to(device), batch_tgt.to(device)
            with torch.no_grad():
                features, att_weight_list = encoder(batch_img)
                part_features, part_att_weights = partlayer(features, att_weight_list)

                cls_features = part_features[:, 0, :]
                logits = classifier(cls_features)
                loss = cross_criterion(logits.squeeze(-1), batch_tgt) + contrastive_loss(cls_features, batch_tgt)
                val_loss += loss.item()
        
        print('Epoch: {}/{}, Val loss: {:.5f}, elpase: {:.3f}s'.format(epoch + 1, n_epochs, val_loss / len(val_dataloader), time.time() - start_time))
        torch.save(encoder.state_dict(), './model_output/transfg_encoder_{}.pt'.format(epoch + 1))
        torch.save(partlayer.state_dict(), './model_output/transfg_partlayer_{}.pt'.format(epoch + 1))
        torch.save(classifier.state_dict(), './model_output/transfg_classifier_{}.pt'.format(epoch + 1))

        val_losses.append(val_loss)

        test_acc = test(encoder, partlayer, classifier, test_dataloader, device)
        test_acc_list.append(test_acc)

        # print(f'att_weights len: {len(part_att_weights)}')
        # for item in part_att_weights:
        #     print(f'    item size: {item.size()}')


    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(range(n_epochs), train_losses)
    line2, = ax1.plot(range(n_epochs), val_losses)
    ax1.set_ylabel('loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('acc')
    line3, = ax2.plot(range(n_epochs), test_acc_list, color=(201 / 255, 1 / 255, 1 / 255))
    plt.legend([line1, line2, line3], ['train loss', 'val loss', 'test acc'])
    plt.xlabel('epoch')
    plt.savefig(f'image_output/transfg_{int(time.time())}.png')
    plt.show()

    return encoder, partlayer, classifier


def test(encoder, partlayer, classifier, test_dataloader, device):
    encoder.to(device)
    partlayer.to(device)
    classifier.to(device)
    encoder.eval()
    partlayer.eval()
    classifier.eval()

    correct_count = 0
    for batch_img_x, batch_target in test_dataloader:
        batch_img_x = batch_img_x.to(device)
      
        with torch.no_grad():
            features, att_weight_list = encoder(batch_img_x)
            part_features, part_att_weights = partlayer(features, att_weight_list)

            cls_features = part_features[:, 0, :]
            logits = classifier(cls_features)
            probs = F.softmax(logits, dim=1)
            predict_indexs = probs.max(dim=1)[1]

            batch_target = torch.tensor(batch_target, dtype=torch.int)
            predict_indexs = predict_indexs.to(torch.device('cpu'))
            for idx, target in zip(predict_indexs, batch_target):
                if target[idx] == 1:
                    correct_count += 1

            # print(f'weight: {part_att_weights.size()}')

    print(f'correct rate: {correct_count / len(test_dataloader)}')

    return correct_count / len(test_dataloader)


if __name__ == '__main__':
    class2idx, train_info, test_info, val_info = make_data(IMAGE_DIR)

    train_dataset = CubDataset(train_info, class2idx, transform=TRANSFORM_TRAIN)
    val_dataset = CubDataset(test_info, class2idx, transform=TRANSFORM_VALID)
    test_dataset = CubDataset(val_info, class2idx, transform=TRANSFORM_VALID)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=DATALOADER_NUM_WORKER, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=DATALOADER_NUM_WORKER, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=True)


    os.makedirs('model_checkpoints', exist_ok=True)
    if not os.path.isfile('model_checkpoints/ViT-B_16-224.npz'):
        urlretrieve('https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz', 'model_checkpoints/ViT-B_16-224.npz')

    config = CONFIGS['ViT-B_16']

    # zero_heads: 是否用0初始化网络权重
    # vis: 似乎是是否返回注意力权重
    model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
    model.load_from(np.load('model_checkpoints/ViT-B_16-224.npz'))

    encoder = nn.Sequential(model.transformer.embeddings, model.transformer.encoder)

    partlayer = PartLayer(config)

    viT_embed_dim = 768
    n_classes = len(class2idx)
    classifier = nn.Linear(viT_embed_dim, n_classes)

    if os.path.exists('model_output/transfg_encoder_20.pt'):
        encoder.load_state_dict(torch.load('model_output/transfg_encoder_20.pt'))
    if os.path.exists('model_output/transfg_classifier_20.pt'):
        classifier.load_state_dict(torch.load('model_output/transfg_classifier_20.pt'))
    if os.path.exists('model_output/transfg_partlayer_20'):
        partlayer.load_state_dict(torch.load('model_output/transfg_partlayer_20'))

    transfg_fine_tune(encoder, partlayer, classifier, train_dataloader, val_dataloader, test_dataloader, NUM_EPOCH, LEARNING_RATE, DEVICE)

    # test(encoder, partlayer, classifier, test_dataloader, DEVICE)
