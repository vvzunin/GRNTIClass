import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):# обычная cross энтропия

    def __init__(self, class_weights= torch.tensor([1.3164, 1.2443, 1.5663, 0.6472, 3.6027, 0.2040, 3.0185, 1.4897, 3.1646,
        0.8802, 1.2860, 2.2852, 0.4165, 0.3968, 1.1249, 0.3536, 1.9390, 0.7354,
        1.8193, 0.3115, 0.8897, 0.2691, 0.6810, 0.6133, 0.6533, 0.7356, 2.0284,
        0.7524, 1.9058, 3.0932, 1.1319, 3.0083], dtype=torch.float64).to('cuda')):
        super().__init__()
        self.xent_loss = nn.BCEWithLogitsLoss(weight=class_weights)

    def forward(self, outputs, targets):
        return self.xent_loss(outputs['predicts'], targets)


class SupConLoss(nn.Module):
# Реализует классы потерь SupConLoss (Supervised Contrastive Loss),
# которая используется в задачах контрастивного обучения с учителем. 
# Основная идея контрастивного обучения заключается в том, 
# чтобы научить модель различать похожие (положительные) и 
# непохожие (отрицательные) примеры в данных. 
# В данном случае функция потерь учитывает метки классов, 
# чтобы улучшить разделение между классами.
    def __init__(self, alpha, temp, class_weights=torch.tensor([1.3164, 1.2443, 1.5663, 0.6472, 3.6027, 0.2040, 3.0185, 1.4897, 3.1646,
        0.8802, 1.2860, 2.2852, 0.4165, 0.3968, 1.1249, 0.3536, 1.9390, 0.7354,
        1.8193, 0.3115, 0.8897, 0.2691, 0.6810, 0.6133, 0.6533, 0.7356, 2.0284,
        0.7524, 1.9058, 3.0932, 1.1319, 3.0083]).to('cuda')):
        super().__init__()
        self.xent_loss = nn.BCEWithLogitsLoss(weight=class_weights)
        self.alpha = alpha
        self.temp = temp


    def remove_diagonal_elements(self, anchor_dot_target):
        """
        Удаляет диагональные элементы из прямоугольной матрицы anchor_dot_target.
        anchor_dot_target: тензор размером [batch_size, num_classes]
        """
        batch_size, num_classes = anchor_dot_target.shape

        # Создаем маску для диагональных элементов
        if batch_size == num_classes:
            # Если матрица квадратная, используем стандартный подход
            mask = torch.eye(batch_size, dtype=torch.bool, device=anchor_dot_target.device)
        else:
            # Если матрица прямоугольная, создаем маску для "диагональных" элементов
            mask = torch.zeros_like(anchor_dot_target, dtype=torch.bool, device=anchor_dot_target.device)
            min_dim = min(batch_size, num_classes)
            mask[:min_dim, :min_dim] = torch.eye(min_dim, dtype=torch.bool, device=anchor_dot_target.device)

        # Обнуляем диагональные элементы
        anchor_dot_target = anchor_dot_target.masked_fill(mask, 0.0)

        return anchor_dot_target

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():

                    # Создаем маску: пары примеров с хотя бы одной общей меткой
            mask = torch.matmul(labels.float(), labels.float().T) > 1e-8  # [batch_size, batch_size]

            # Удаляем диагональные элементы (пары с самим собой)
            mask = mask ^ torch.diag_embed(torch.diag(mask))


        anchor_dot_target = torch.einsum('bd,cd->bc',#'bnd,bmd->bnm',#'bd,cd->bc', 
                                         anchor, target) / self.temp
        
        # print(anchor_dot_target.shape)

        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        
        # self.remove_diagonal_elements(anchor_dot_target)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target,
                                  dim=1, keepdim=True)
        # print
        
        logits = anchor_dot_target - logits_max.detach()
        # compute log prob
        exp_logits = torch.exp(logits)
        # mask out positives


        logits = logits * mask

        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        # in case that mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss


class DualLoss(SupConLoss):# двойной
# Класс DualLoss комбинирует три компонента потерь:

# Кросс-энтропийная потеря:
# Обучает модель правильно классифицировать данные.

# Контрастивная потеря (1):
# Улучшает представление признаков классов, сближая их с соответствующими признаками меток.

# Контрастивная потеря (2):
# Улучшает представление признаков меток, сближая их с соответствующими признаками классов.
    def __init__(self, alpha, temp):
        super().__init__(alpha, temp)

    def forward(self, outputs, targets):
        # print("Shape of cls_feats:", len(outputs['cls_feats']))
        # print(outputs['cls_feats'])

        # print(outputs['cls_feats'])

        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)


        # normed_pos_label_feats = torch.gather(normed_label_feats, dim=1,
        #                                       index=targets.reshape(-1, 1, 1
        #                                                             ).expand(-1, 1, 
        #                                                                      normed_label_feats.size(-1))).squeeze(1)
        
        # Для многозадачной классификации: 
        # получаем среднее представление для всех активных меток

        # Преобразуем targets в индексы активных меток
        active_labels = targets.unsqueeze(-1)  # [batch_size, num_classes, 1]

        # Умножаем признаки на бинарную маску активных меток
        normed_pos_label_feats = normed_label_feats * active_labels  # [batch_size, num_classes, feature_dim]

        # Суммируем признаки по всем активным меткам
        normed_pos_label_feats = normed_pos_label_feats.sum(dim=1)  # [batch_size, feature_dim]

        # Нормализуем результат
        normed_pos_label_feats = F.normalize(normed_pos_label_feats, dim=-1)

        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss_1 = 0.5 * self.alpha * self.nt_xent_loss(normed_pos_label_feats,
                                                         normed_cls_feats, targets)
        cl_loss_2 = 0.5 * self.alpha * self.nt_xent_loss(normed_cls_feats,
                                                         normed_pos_label_feats, targets)
        return ce_loss + cl_loss_1 + cl_loss_2
