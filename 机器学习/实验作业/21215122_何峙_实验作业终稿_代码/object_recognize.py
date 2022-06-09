from configs import *
from models.modeling import VisionTransformer, CONFIGS
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn
import torch.nn.functional as F


class ObjectFinder():

    def __init__(self, 
                 encoder, 
                 classifier, 
                 detect_threshold=0.1,
                 reconsider_threshold=0.001,
                 max_iter=5,
                 image_transform=None):
        self.encoder = encoder
        self.encoder.eval()
        self.classifier = classifier
        self.classifier.eval()
        self.detect_threshold = detect_threshold
        self.reconsider_threshold = reconsider_threshold
        self.max_iter = max_iter

        if image_transform:
            self.transform = image_transform
        else:
            self.transform = transforms.ToTensor()

    def extract_CLS_final_layer_att_mat(self, att_mat):
        att_mat = torch.stack(att_mat).squeeze(1) # [12 (layers), 12 (heads), 197, 197]
        att_mat = torch.mean(att_mat, dim=1)      # [12, 197, 197]
        residual_att = torch.eye(att_mat.size(1)) # [197, 197]
        aug_att_mat = att_mat + residual_att      # [12, 197, 197]
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1) # normalize
 
        grid_size = int(np.sqrt(aug_att_mat.size(-1)-1)) # 有一个是用于分类的CLS token，所以-1 
 
        joint_attentions = torch.zeros(aug_att_mat.size())  # [12, 197, 197]
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

        att_last_layer = joint_attentions[-1]  # [197, 197]
        att_CLS = att_last_layer[0, 1:]        # [196]， 第一行就是CLS token对应于各个token的attention权重，取“1:”就是与其他token的attention权重
        att_CLS = att_CLS.reshape(grid_size, grid_size).detach().numpy() # [14, 14]
        att_CLS = att_CLS / att_CLS.max()  # normalize

        print(f'grid size: {grid_size}')
        print(f'joint_attentions: {joint_attentions.size()}')

        return att_CLS

    def mask_image(self, img_path, plot=True, labels=None, topN=3):
        im = Image.open(img_path).convert('RGB') 
        im_x = self.transform(im).unsqueeze(0) 

        with torch.no_grad():
            features, att_mat = self.encoder(im_x)       # att_mat是注意力权重矩阵：12个 [1, 12, 197, 197] tensors
        logits = self.classifier(features[:, 0, :]) 
        probs = nn.Sigmoid()(logits).squeeze(0) 

        att_CLS = self.extract_CLS_final_layer_att_mat(att_mat)
    

        mask = cv2.resize(att_CLS, (np.array(im).shape[0], np.array(im).shape[1]))
        if (mask.shape[0]==np.array(im).shape[1]) and (mask.shape[1]==np.array(im).shape[0]):
            mask = cv2.resize(mask, (np.array(im).shape[1], np.array(im).shape[0]))   
    
        im_ = np.array(im).transpose(2,0,1)
        im_masked_new = np.array([im_channel * mask for im_channel in im_])
        im_masked_new = im_masked_new / im_masked_new.max()  # normalize

        if labels:
            top_cls_idxs = torch.argsort(probs, dim=-1, descending=True)
            labels = [l + "\n" for l in labels] 
            if plot:
                for idx in top_cls_idxs[:topN]:
                    print(f'{probs[idx.item()]:.5f} : {labels[idx.item()]}', end='')


        if not os.path.exists("mask_images"):
            os.makedirs("mask_images")
        fname = "mask_images/" + img_path.split("/")[-1] + "_masked.jpg"
    
        plt.imsave(fname, im_masked_new.transpose(1,2,0))

        if plot:
            plt.imshow(im_masked_new.transpose(1,2,0))
            plt.title("Masked Image:")

            mask = cv2.normalize(mask[..., np.newaxis], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # [im_H, im_W]
            mask_heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET) # [im_H, im_W, 3]
            im_temp = cv2.imread(img_path, 1) 

            overlaid = cv2.addWeighted(im_temp, 0.5, mask_heatmap, 0.5, 0)
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))
            ax1.set_title('Original')
            ax2.set_title('Heatmap')
            ax3.set_title('Attention Map')
            _ = ax1.imshow(im)
            _ = ax2.imshow(mask_heatmap[:,:,::-1])
            _ = ax3.imshow(overlaid[:,:,::-1])
            # plt.show()
    
        return im_masked_new, probs, fname


    def detect_class(self, img_path):
        fname = img_path
        iter = 0
        while (iter < self.max_iter): 
            im_masked, probs, fname = self.mask_image(fname, plot=False)

            reconsider = [int(p >= self.reconsider_threshold) for p in probs] 
            if im_masked is None or sum(reconsider) == 0:
                return 0 

            detected = [int(p >= self.detect_threshold) for p in probs] 
            if sum(detected) == 1:  # 检测到一个类别
                return detected.index(1) + 1
            elif sum(detected) > 1: # 检测到多个类别
                return [i + 1 for i, d in enumerate(detected) if d == 1]

            iter += 1

        return 0 



if __name__ == '__main__':

    config = CONFIGS['ViT-B_16']    # ViT-B_16，有12层transfomer_block，每个transformer有12个attention heads

    # zero_heads: 是否用0初始化网络权重
    # vis: 似乎是是否返回注意力权重
    model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
    model.load_from(np.load('model_checkpoints/ViT-B_16-224.npz'))

    encoder = nn.Sequential(model.transformer.embeddings,
                            model.transformer.encoder)
    ViT_embed_dim = 768
    n_classes = 200
    classifier = nn.Linear(ViT_embed_dim, n_classes)

    if os.path.exists('model_output/transfg_encoder_20.pt'):
        encoder.load_state_dict(torch.load('model_output/transfg_encoder_20.pt'))
    if os.path.exists('model_output/transfg_classifier_20.pt'):
        classifier.load_state_dict(torch.load('model_output/transfg_classifier_20.pt'))

    # image_path = '../data/cub_200_2011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg'
    # image_path = '../data/cub_200_2011/CUB_200_2011/images/006.Least_Auklet/Least_Auklet_0030_795116.jpg'
    image_path = '../data/cub_200_2011/CUB_200_2011/images/113.Baird_Sparrow/Baird_Sparrow_0032_794553.jpg'
    image_path1 = '../data/cub_200_2011/CUB_200_2011/images/032.Mangrove_Cuckoo/Mangrove_Cuckoo_0011_26406.jpg'

    finder = ObjectFinder(encoder, classifier, image_transform=TRANSFORM_VALID)
    _, prob, _ = finder.mask_image(img_path=image_path1, plot=True)


    # im = Image.open(image_path).convert('RGB')
    # transform = TRANSFORM_VALID
    # im_x = transform(im)            # [1, 3, 224, 224]， patch size: [16, 16]

    # im1 = Image.open(image_path1).convert('RGB')
    # im_x1 = transform(im1)           # [1, 3, 224, 224]， patch size: [16, 16]
    # batch_x = torch.stack((im_x, im_x1))
    # print(f'batch size: {batch_x.size()}')

    # print(f'image tensor size: {im_x.size()}')
    # print(f'image shape: {np.array(im).shape}')

    # features, att_mat = encoder(batch_x)            
    # att_mat = torch.stack(att_mat).squeeze(1)    # [12 (layers), 12 (heads), 197, 197]
    # att_mat = torch.mean(att_mat, dim=1)         # [12, 197, 197]
    # residual_att = torch.eye(att_mat.size(1))    # [197, 197]
    # aug_att_mat = att_mat + residual_att         # [12, 197, 197]
    # aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)    # [12, 197, 197]
    # grid_size = int(np.sqrt(aug_att_mat.size(-1) - 1))   # 有一个是用于分类的CLS token

    # joint_attentions = torch.zeros(aug_att_mat.size())    # [12, 197, 197]
    # joint_attentions[0] = aug_att_mat[0]
    # for n in range(1, aug_att_mat.size(0)):
    #     joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # att_last_layer = joint_attentions[-1]  # [197, 197]
    # att_CLS = att_last_layer[0, 1:]        # [197]，第一行就是CLS token对应于各个token的attention权重，取“1:”就是与其他token的attention权重
    # att_CLS = att_CLS.reshape(grid_size, grid_size).detach().numpy()   # [14, 14]
    # att_CLS = att_CLS / att_CLS.max() 

    # print(features.size(), aug_att_mat.size())

    # print(f'feature size: {features.size()}')
    # print(f'att mat size: {att_mat[0].size()}')

    #  # att_mat为 12(层) x [batch_size, 12(head), 197, 197]
    # joint_att_mat = att_mat[0]   # [batch_size, 12(head_size), 197, 197]
    # for n in range(1, len(att_mat)):
    #     joint_att_mat = torch.matmul(att_mat[n], joint_att_mat)

    # print(joint_att_mat.size())
    # joint_att_mat = joint_att_mat[:, :, 0, 1:]  # [batch_size, head_size, 196]，找CLS token attent到其他token的注意力
    # print(joint_att_mat.size())
    # print(joint_att_mat.max(2)[1])

    # att_index = joint_att_mat.max(2)[1] + 1
    # parts = []
    # for batch in range(att_index.shape[0]):
    #     parts.append(features[batch, att_index[batch, :], :])
    # parts = torch.stack(parts).squeeze(1)
    # print(f'parts size: {parts.size()}')
    # print(f'cls feature: {features[:, 0].unsqueeze(1).size()}')
    # concat = torch.cat((features[:, 0].unsqueeze(1), parts), dim=1)
    # print(f'concat: {concat.size()}')
    # print(f'cls_embedding: {concat[:, 0].size()}')

    # features = concat[:, 0]

    # B, _ = features.shape
    # features = F.normalize(features)
    # cos_matrix = features.mm(features.t())
    # print(cos_matrix)
    # targets = torch.tensor([[1, 0, 0], [1, 0, 0]])
    # pos_label_matrix = targets.mm(targets.t())
    # neg_label_matrix = 1 - pos_label_matrix
    # pos_cos_matrix = 1 - cos_matrix
    # print(pos_label_matrix)
    # print((pos_cos_matrix * pos_label_matrix).sum())
