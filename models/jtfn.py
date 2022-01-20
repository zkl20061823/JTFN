import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.base import get_base
from models.backbone.resbase import get_resbase
from models.modules import FIM, GAU

class JTFN(nn.Module):
    def __init__(self, backbone, use_gau, use_fim, up, classes=1, steps=3, reduce_dim=False):
        super(JTFN, self).__init__()
        assert backbone in ['base64', 'base32', 'resbase32', 'resbase64']
        assert classes == 1
        assert len(use_gau) == 5
        assert len(use_fim) == 4
        self.backbone = backbone
        self.use_gau = use_gau
        self.use_fim = use_fim
        self.steps = steps
        self.up = up
        self.reduce_dim = reduce_dim
        self.bce_loss = nn.BCELoss()

        if self.backbone == 'base64':
            print('INFO: Using base backbone')
            filters = [64, 128, 256, 512, 512]
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_base(filters)
        elif self.backbone == 'base32':
            print('INFO: Using base backbone')
            filters = [32, 64, 128, 256, 512]
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_base(filters)
        elif self.backbone == 'resbase32':
            print('INFO: Using base backbone')
            filters = [32, 64, 128, 256, 512]
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_resbase(filters)
        elif self.backbone == 'resbase64':
            print('INFO: Using base backbone')
            filters = [64, 128, 256, 512, 512]
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_resbase(filters)
        else:
            raise RuntimeError('Backbone ', backbone, 'is not implemented.')

        if reduce_dim:
            reduce_filters = [32, 32, 32, 32, 32]
        else:
            reduce_filters = filters

        self.skip_blocks = []
        for i in range(5):
            self.skip_blocks.append(GAU(filters[i], use_gau[i], reduce_dim, reduce_filters[i]))
        self.decoder = []
        index = use_fim.index(False) if False in use_fim else len(use_fim) - 1
        for i in range(4):
            if i == (index-1):
                self.decoder.append(FIM(reduce_filters[i+1], reduce_filters[i], reduce_filters[i], use_fim[i], up[i], bottom=True))
            else:
                self.decoder.append(FIM(reduce_filters[i + 1], reduce_filters[i], reduce_filters[i], use_fim[i], up[i]))
        self.skip_blocks = nn.ModuleList(self.skip_blocks)
        self.decoder = nn.ModuleList(self.decoder)

        self.filters = filters
        self.reduce_filters = reduce_filters

    def forward(self, batch_dict):
        img = batch_dict['img']
        bs, c, h, w = img.shape

        output_dict = {}
        y = torch.zeros([bs, 1, h, w], device=img.device)
        bc_x1 = self.layer0(img)
        bc_x2 = self.layer1(bc_x1)
        bc_x3 = self.layer2(bc_x2)
        bc_x4 = self.layer3(bc_x3)
        bc_x5 = self.layer4(bc_x4)

        for i in range(self.steps):
            x1 = self.skip_blocks[0](bc_x1, y)
            x2 = self.skip_blocks[1](bc_x2, y)
            x3 = self.skip_blocks[2](bc_x3, y)
            x4 = self.skip_blocks[3](bc_x4, y)
            x5 = self.skip_blocks[4](bc_x5, y)

            x5_s = x5
            x5_b = x5
            x4_s, x4_b, s4_cls, b4_cls = self.decoder[-1](x5_s, x5_b, x4)
            x3_s, x3_b, s3_cls, b3_cls = self.decoder[-2](x4_s, x4_b, x3)
            x2_s, x2_b, s2_cls, b2_cls = self.decoder[-3](x3_s, x3_b, x2)
            x1_s, x1_b, s1_cls, b1_cls = self.decoder[-4](x2_s, x2_b, x1)

            output_dict['step_' + str(i) + '_output_mask'] = [s1_cls, s2_cls, s3_cls, s4_cls]
            output_dict['step_' + str(i) + '_output_boundary'] = [b1_cls, b2_cls, b3_cls, b4_cls]
            y = s1_cls

        output_dict['output'] = y
        return output_dict


    def compute_objective(self, output_dict, batch_dict, multi_layer=True):
        loss_dict = {}

        gt_mask = batch_dict['anno_mask']
        gt_boundary = batch_dict['anno_boundary']
        h, w = gt_mask.shape[2], gt_mask.shape[3]

        total_loss = None
        for i in range(self.steps):
            pred_mask = output_dict['step_' + str(i) + '_output_mask'] # list
            pred_boundary = output_dict['step_' + str(i) + '_output_boundary'] # list
            step_mask_loss = None
            if multi_layer:
                for k in range(len(pred_mask)):
                    inner_pred = pred_mask[k]
                    if inner_pred is None:
                        continue
                    inner_pred = F.interpolate(inner_pred, (h, w), mode='bilinear', align_corners=True)
                    mask_loss = self.bce_loss(inner_pred, gt_mask.float())
                    if step_mask_loss is None:
                        step_mask_loss = torch.zeros_like(mask_loss).to(mask_loss.device)
                    step_mask_loss = step_mask_loss + mask_loss
                step_mask_loss = step_mask_loss / len(pred_mask)
            else:
                inner_pred = pred_mask[0]
                inner_pred = F.interpolate(inner_pred, (h, w), mode='bilinear', align_corners=True)
                mask_loss = self.bce_loss(inner_pred, gt_mask.float())
                step_mask_loss = mask_loss

            step_topo_loss = None
            if multi_layer:
                for k in range(len(pred_boundary)):
                    inner_pred = pred_boundary[k]
                    if inner_pred is None:
                        continue
                    inner_pred = F.interpolate(inner_pred, (h, w), mode='bilinear', align_corners=True)
                    topo_loss = self.bce_loss(inner_pred, gt_boundary.float())
                    if step_topo_loss is None:
                        step_topo_loss = torch.zeros_like(topo_loss).to(topo_loss.device)
                    step_topo_loss = step_topo_loss + topo_loss
                if step_topo_loss is not None:
                    step_topo_loss = step_topo_loss / len(pred_boundary)
                else:
                    step_topo_loss = torch.zeros_like(step_mask_loss).to(step_mask_loss.device)
            else:
                inner_pred = pred_boundary[0]
                inner_pred = F.interpolate(inner_pred, (h, w), mode='bilinear', align_corners=True)
                topo_loss = self.bce_loss(inner_pred, gt_boundary.float())
                step_topo_loss = topo_loss

            if total_loss is None:
                total_loss = torch.zeros_like(step_mask_loss).to(step_mask_loss.device)
            total_loss = total_loss + step_mask_loss + step_topo_loss

            loss_dict['step_' + str(i) + 'total_loss'] = step_mask_loss + step_topo_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict

    def train_mode(self):
        self.train()

    def test_mode(self):
        self.eval()
        self.layer0.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        self.layer4.eval()