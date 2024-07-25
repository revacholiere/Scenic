import logging

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VisionTransformer
from torch.functional import F
from config import cfg
import torch.distributions as dist


logger = logging.getLogger(__name__)


# Vision transformer as encoder
class PretrainedVisionTransformer(nn.Module):
    def __init__(self):
        super(PretrainedVisionTransformer, self).__init__()
        self.vit = self.vit = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        )

    def forward(self, x: torch.Tensor):
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, image_size, patch_size):
        super(VisionTransformer, self).__init__()

        self.patch_embed = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (image_size // patch_size) ** 2
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, hidden_size)
        )
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([x, self.positional_embedding.repeat(x.shape[0], 1, 1)], dim=1)
        x = self.encoder(x)
        # import pdb
        # pdb.set_trace()
        # return x[:, 0]
        return x


# Caption based Transformer as decoder
class CaptionDecoder(nn.Module):
    def __init__(
        self, hidden_size, num_heads, num_layers, vocabulary_size, max_caption_length
    ):
        super(CaptionDecoder, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_caption_length, hidden_size)
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, num_heads), num_layers
        )
        self.fc = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, encoder_output, captions):
        embeddings = self.embedding(captions)

        # Apply positional encoding
        embeddings *= torch.sqrt(torch.FloatTensor([self.embedding.embedding_dim])).to(
            captions.device
        )
        embeddings += self.positional_encoding[:, : captions.shape[1]].to(
            captions.device
        )

        # Permute to match Transformer input format
        embeddings = embeddings.permute(1, 0, 2)
        encoder_output = encoder_output.permute(1, 0, 2)

        # Generate target mask
        device = captions.device  # Get the device of the captions tensor
        tgt_mask = torch.triu(
            torch.ones(captions.shape[1], captions.shape[1], device=device), diagonal=1
        ).bool()

        # Apply Transformer Decoder
        outputs = self.decoder(tgt=embeddings, memory=encoder_output, tgt_mask=tgt_mask)

        # Apply linear layer for vocabulary prediction
        outputs = self.fc(outputs.permute(1, 0, 2))

        return outputs


# Transformer based encoder decoder
class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        num_classes,
        image_size,
        patch_size,
        max_caption_length,
    ):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = VisionTransformer(
            hidden_size, num_heads, num_encoder_layers, image_size, patch_size
        )
        self.decoder = CaptionDecoder(
            hidden_size,
            num_heads,
            num_decoder_layers,
            num_classes,
            max_caption_length,
        )

    def forward(self, images, captions):
        encoder_output = self.encoder(images)
        decoder_output = self.decoder(encoder_output, captions)
        return decoder_output

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


class PretrainedImageCaptioningModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        num_classes,
        image_size,
        patch_size,
        max_caption_length,
    ):
        super(PretrainedImageCaptioningModel, self).__init__()

        self.encoder = PretrainedVisionTransformer()
        self.decoder = CaptionDecoder(
            hidden_size,
            num_heads,
            num_decoder_layers,
            num_classes,
            max_caption_length,
        )

    def forward(self, images, captions):
        encoder_output = self.encoder(images)
        decoder_output = self.decoder(encoder_output, captions)
        return decoder_output

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


def get_pretrain_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.common.transform_normalize_mean,
                std=cfg.common.transform_normalize_std,
            ),
        ]
    )


def get_train_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (cfg.common.img_width_resize, cfg.common.img_height_resize),
                antialias=True,
            ),
            transforms.Normalize(
                mean=cfg.common.transform_normalize_mean,
                std=cfg.common.transform_normalize_std,
            ),
        ]
    )
def calculate_iou(bbox1, bbox2):
    # Calculate intersection coordinates
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[2], bbox2[2])
    x_right = min(bbox1[1], bbox2[1])
    y_bottom = min(bbox1[3], bbox2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas of both bounding boxes
    bbox1_area = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2])
    bbox2_area = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2])
    
    # Calculate union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IoU
    if union_area != 0:
        iou = intersection_area / union_area
    else:
        iou = 0
    return iou

def compute_recall(bboxes, bboxes_gt, labels,labels_gt,iou_threshold=0.5):
    true_positives = 0
    false_negatives = 0
    recall = 0
    for bbox,label in zip(bboxes,labels):
        bbox_is_tp = False
        for bbox_gt,label_gt in zip(bboxes_gt,labels_gt):
            vehicle_list = ['motorcycle','truck','bicycle','bus','car']
            if label in vehicle_list:
                label = 'car'
            if label != label_gt:
                continue  # Labels don't match, skip
            iou = calculate_iou(bbox, bbox_gt)
            # print(f'iou is {iou}, bbox is {label} {bbox}, bbox gt is {label_gt} {bbox_gt}')
            if iou >= iou_threshold:
                bbox_is_tp = True
                break
        if label in ['car','person']:
            if bbox_is_tp:
                true_positives += 1
            else:
                false_negatives += 1
    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    return recall

def compute_matchness(bboxes, bboxes_gt, labels,labels_gt,iou_threshold=0.5):
    reward = []
    for bbox,label in zip(bboxes,labels):
        bbox_is_tp = False
        for bbox_gt,label_gt in zip(bboxes_gt,labels_gt):
            vehicle_list = ['motorcycle','truck','bicycle','bus','car']
            if label in vehicle_list:
                label = 'car'
            if label != label_gt:
                continue  # Labels don't match, skip
            iou = calculate_iou(bbox, bbox_gt)
            if iou >= iou_threshold:
                bbox_is_tp = True
                break
        if bbox_is_tp:
            reward.append(1)
        else:
            reward.append(-1)
    return reward

def bbox_resize_xxyy(bbox):
    xmin = bbox[0,0]
    ymin = bbox[0,1]
    xmax = bbox[1,0]
    ymax = bbox[1,1]
    width_ratio = cfg.carla.ego_camera.image_width / cfg.common.img_width_resize
    height_ratio = cfg.carla.ego_camera.image_height / cfg.common.img_height_resize
    xmin_resize = xmin / width_ratio
    ymin_resize = ymin / height_ratio
    xmax_resize = xmax / width_ratio
    ymax_resize = ymax / height_ratio
    xmin_resize = max(1, xmin_resize)
    ymin_resize = max(1, ymin_resize)
    xmax_resize = min(cfg.common.img_width_resize, xmax_resize)
    ymax_resize = min(cfg.common.img_height_resize, ymax_resize)

    return [int(xmin_resize),int(xmax_resize),int(ymin_resize),int(ymax_resize)]

def predict_detections(model, device, image, vocabulary, sample=False):
    with torch.no_grad():
        encoder_output = model.encoder(image)
        # start_token = torch.tensor(
        #     [vocabulary.get_index_from_word(vocabulary.start_token)]
        # ).view(
        #     1, 1
        # )  # Index of the start token in your vocabulary
        maxWH = max(cfg.common.img_width_resize,cfg.common.img_height_resize)
        # current_token = start_token
        # generated_caption = [vocabulary.get_index_from_word(vocabulary.start_token)]

        ar_outputs = [vocabulary.get_index_from_word(vocabulary.start_token)]

        ar_outputs = torch.tensor(ar_outputs)
        ar_outputs = ar_outputs.unsqueeze(1)
        ar_outputs = ar_outputs.to(device)

        for i in range(cfg.common.max_caption_length):
            # Pass the entire generated caption to the decoder
            # current_token = current_token.to(device)
            # decoder_output = model.decoder(encoder_output, current_token)
            decoder_output = model.decoder(encoder_output, ar_outputs)
            prob = F.softmax(decoder_output, dim=-1)

            probs = prob
            if i % 5 == 0:
                indices_to_mask = [0,1,3] + list(range(4,84))
                probs[:,:,indices_to_mask] = 0
                probs_flat = probs.view(-1, probs.size(-1))
                sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
                sampled_indices = sampled_indices_flat.view(probs.size()[:-1])
                if sampled_indices[:,-1].item() == 2:
                    break
                # logger.info(sampled_indices[:,-1])
                # logger.info(sampled_indices[:,-1].shape)
                # logger.info(sampled_indices[:,-1].item())

            if i % 5 == 2:
                indices_to_mask = [0,1,2,3] + list(range(4,84))
                probs[:,:,indices_to_mask] = 0
                probs_flat = probs.view(-1, probs.size(-1))
                sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
                sampled_indices = sampled_indices_flat.view(probs.size()[:-1])
            
            if i % 5 == 1 or i % 5 == 3:
                indices_to_mask = [0,1,2,3] + list(range(4,84)) 
                probs[:,:,indices_to_mask] = 0
                indices_to_mask = ar_outputs[:,-1]

                mask = torch.ones_like(probs)
                for j in range(len(indices_to_mask)):
                    mask[:, :, j][mask[:, :, j] <= indices_to_mask[j]] = 0
                probs = probs * mask
                probs_flat = probs.view(-1, probs.size(-1))
                sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
                sampled_indices = sampled_indices_flat.view(probs.size()[:-1])

            if i % 5 == 4:
                indices_to_mask = [0,1,2,3] + list(range(84,84+maxWH))
                probs[:,:,indices_to_mask] = 0
                probs_flat = probs.view(-1, probs.size(-1))
                sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
                sampled_indices = sampled_indices_flat.view(probs.size()[:-1])
            ar_outputs = torch.cat((ar_outputs,sampled_indices[:,-1].unsqueeze(1)),dim=1)
            # sorted_prob,indices = torch.sort(prob,dim=2,descending=True)
            # # logger.info(f"sorted_prob shape is{sorted_prob.shape}, indices shape is {indices.shape}")
            # prob_last,index_last = sorted_prob[:,-1,:],indices[:,-1,:]
            # index_dist = dist.Categorical(prob_last)
            # sample_index = index_dist.sample()
            # next_token = decoder_output.argmax(dim=2)[
            #     -1
            # ]  # Select the token with the highest probability
            # logger.info(f"sampled avlue is {index_last[0,sample_index].item()}")
            # logger.info(f"highest token is {index_last[0,0].item()}")
            # logger.info(f"next token is {next_token}")
            # if index_last[0,sample_index].item() == vocabulary.get_index_from_word(
            #     vocabulary.end_token
            # ):
            #     break
            # generated_caption.append(index_last[0,sample_index].item())
            # if next_token[-1].item() == vocabulary.get_index_from_word(
            #     vocabulary.end_token
            # ):  # Index of the end token in your vocabulary
            #     break
            # generated_caption.append(next_token[-1].item())

            # Update the current token to include the newly generated token
            # current_token = torch.tensor(generated_caption).view(1, -1)
        generated_caption = ar_outputs.reshape(ar_outputs.shape[1],).tolist()
    # Convert the generated caption to words
    # logger.info(generated_caption)
    # logger.info(ar_outputs)
    generated_caption_words = [
        vocabulary.get_word_from_index(idx) for idx in generated_caption
    ]
    logger.debug(f"Generated caption: {generated_caption_words}")
    return (
        convert_tokens_to_detections(generated_caption_words, vocabulary),
        generated_caption_words,
        generated_caption,
    )

def get_bboxes_and_labels(indices,idx_to_word):
    bboxes_str =[idx_to_word[index] for index in indices]
    bboxes = []
    labels = []
    num_obj = int((len(bboxes_str))/5)
    for j in range(num_obj):
        bbox = []
        if bboxes_str[j*5] == '<end>':
            break
        
        if bboxes_str[j*5].isdigit() and bboxes_str[j*5+1].isdigit() and bboxes_str[j*5+2].isdigit() and bboxes_str[j*5+3].isdigit():
            bbox.append(int(bboxes_str[j*5]))
            bbox.append(int(bboxes_str[j*5+1]))
            bbox.append(int(bboxes_str[j*5+2]))
            bbox.append(int(bboxes_str[j*5+3]))
        else:
            continue
        label = bboxes_str[j*5+4]
        if bbox[0] > bbox[1] or bbox[2] > bbox[3]:
            continue
        bboxes.append(bbox)
        labels.append(label)
    return bboxes,labels


def convert_bbox_xywh_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def convert_bbox_xyxy_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def convert_bboxes_xywh_xyxy(bboxes):
    return [convert_bbox_xywh_xyxy(bbox) for bbox in bboxes]


def convert_bboxes_xyxy_xywh(bboxes):
    return [convert_bbox_xyxy_xywh(bbox) for bbox in bboxes]


def convert_bbox_xxyy_xyxy(bbox):
    x1, x2, y1, y2 = bbox
    return [x1, y1, x2, y2]


def convert_bbox_xyxy_xxyy(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, x2, y1, y2]


def resize_bbox_xyxy(bbox, source_size, target_size):
    source_width, source_height = source_size
    target_width, target_height = target_size
    x1, y1, x2, y2 = bbox
    x1 = x1 * target_width / source_width
    y1 = y1 * target_height / source_height
    x2 = x2 * target_width / source_width
    y2 = y2 * target_height / source_height
    return (x1, y1, x2, y2)


def resize_bboxes_xyxy(bboxes, source_size, target_size):
    return [resize_bbox_xyxy(bbox, source_size, target_size) for bbox in bboxes]


def resize_bbox_xywh(bbox, source_size, target_size):
    source_width, source_height = source_size
    target_width, target_height = target_size
    x, y, w, h = bbox
    x = x * target_width / source_width
    y = y * target_height / source_height
    w = w * target_width / source_width
    h = h * target_height / source_height
    return (x, y, w, h)


def resize_bboxes_xywh(bboxes, source_size, target_size):
    return [resize_bbox_xywh(bbox, source_size, target_size) for bbox in bboxes]


def convert_tokens_to_detections(tokens, vocabulary):
    if vocabulary.start_token in tokens:
        tokens = tokens[tokens.index(vocabulary.start_token) + 1 :]
    bboxes_xyxy = []
    categories = []
    confidences = []  # All confidences are 1 in this case

    # Ensure the tokens list length is a multiple of 5 by trimming any extra tokens
    num_chunks = len(tokens) // 5
    tokens = tokens[: num_chunks * 5]

    for i in range(0, len(tokens), 5):
        chunk = tokens[i : i + 5]
        try:
            # Attempt to convert the bounding box coordinates and class label to floats/integers
            bbox_xxyy = [int(coord) for coord in chunk[:4]]
            bbox_xyxy = convert_bbox_xxyy_xyxy(bbox_xxyy)
            category_label = chunk[4]
            # skip detections with invalid bounding boxes
            if bbox_xyxy[0] >= bbox_xyxy[2] or bbox_xyxy[1] >= bbox_xyxy[3]:
                continue
            # Append the data to the output lists if the conversion is successful
            bboxes_xyxy.append(bbox_xyxy)
            categories.append(category_label)
            confidences.append(1)  # Confidence is 1 for all entries
        except ValueError as e:
            # Skip this chunk if any conversion fails
            continue

    return bboxes_xyxy, categories, confidences


def convert_detections_to_tokens(bboxes_xyxy, categories, confidences, vocabulary):
    # confidences are ignored for now

    tokens = []
    for bbox_xyxy, category_label in zip(bboxes_xyxy, categories):
        bbox_xyxy_int = list(map(int, bbox_xyxy))
        # filter the inconsistent box
        if bbox_xyxy_int[0] > bbox_xyxy_int[2] or bbox_xyxy_int[1] > bbox_xyxy_int[3]:
            continue
        if bbox_xyxy_int[0] > cfg.common.img_width_resize:
            continue
        if bbox_xyxy_int[1] > cfg.common.img_height_resize:
            continue
        if bbox_xyxy_int[2] < 1:
            continue
        if bbox_xyxy_int[3] < 1:
            continue            
        if bbox_xyxy_int[0] < 1:
            bbox_xyxy_int[0] = 1
        if bbox_xyxy_int[1] < 1:
            bbox_xyxy_int[1] = 1
        if bbox_xyxy_int[2] > cfg.common.img_width_resize:
            bbox_xyxy_int[2] = cfg.common.img_width_resize
        if bbox_xyxy_int[3] > cfg.common.img_height_resize:
            bbox_xyxy_int[3] = cfg.common.img_height_resize
        # bbox_xxyy = convert_bbox_xyxy_xxyy(bbox_xyxy)
        tokens += [str(val) for val in bbox_xyxy_int]
        tokens.append(str(category_label))
    tokens = [vocabulary.start_token] + tokens + [vocabulary.end_token]
    return tokens


def filter_detections_by_categories(
    bboxes, categories, confidences, category_whitelist
):
    filtered_bboxes = []
    filtered_categories = []
    filtered_confidences = []
    for bbox, category_label, confidence in zip(bboxes, categories, confidences):
        if category_label in category_whitelist:
            filtered_bboxes.append(bbox)
            filtered_categories.append(category_label)
            filtered_confidences.append(confidence)
    return filtered_bboxes, filtered_categories, filtered_confidences


def replace_categories(categories, categories_to_replace, new_category):
    return [
        new_category if category_label in categories_to_replace else category_label
        for category_label in categories
    ]


def compare_models(model1, model2):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    def log_diff(message, diff):
        if diff:
            logger.warning(message)
        else:
            logger.debug(message)

    # Compare keys (this checks if they have the same architecture)
    architecture_different = state_dict1.keys() == state_dict2.keys()
    log_diff(
        f"Architecture different?: {architecture_different}", architecture_different
    )
    # Compare values
    for (key1, tensor1), (key2, tensor2) in zip(
        state_dict1.items(), state_dict2.items()
    ):
        tensor_different = torch.equal(tensor1, tensor2)
        log_diff(f"Tensor {key1} different?: {tensor_different}", tensor_different)

    for (key1, tensor1), (key2, tensor2) in zip(
        state_dict1.items(), state_dict2.items()
    ):
        diff = torch.norm(tensor1 - tensor2)
        logger.debug(f"L2 Norm {key1}: {diff.item()}")
