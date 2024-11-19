import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
import gc

def clear_gpu_memory():
    """Clear GPU memory cache"""
    torch.cuda.empty_cache()
    gc.collect()

def test(obj_names, mvtec_path, checkpoint_path, base_model_name):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    
    # Set memory efficient options
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    for obj_name in obj_names:
        clear_gpu_memory()
        
        img_dim = 256
        run_name = base_model_name+"_"+obj_name+'_'

        # Initialize models
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

        try:
            model.load_state_dict(
                torch.load(
                    os.path.join(checkpoint_path, run_name+".pckl"),
                    map_location='cuda:0',
                    weights_only=True
                )
            )
            model_seg.load_state_dict(
                torch.load(
                    os.path.join(checkpoint_path, run_name+"_seg.pckl"),
                    map_location='cuda:0',
                    weights_only=True
                )
            )
        except Exception as e:
            print(f"Error loading model for {obj_name}: {str(e)}")
            continue

        model.cuda()
        model.eval()
        model_seg.cuda()
        model_seg.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        pixel_scores = []
        gt_pixel_scores = []
        anomaly_score_gt = []
        anomaly_score_prediction = []

        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch % 10 == 0:
                clear_gpu_memory()

            try:
                with torch.amp.autocast('cuda'):  # Updated autocast syntax
                    with torch.no_grad():
                        gray_batch = sample_batched["image"].cuda()
                        is_normal = sample_batched["has_anomaly"].detach().cpu().numpy()[0, 0]
                        anomaly_score_gt.append(is_normal)
                        
                        true_mask = sample_batched["mask"]
                        true_mask_cv = true_mask.detach().cpu().numpy()[0, :, :, :].transpose((1, 2, 0))

                        # Get reconstructions - now returns (output, features)
                        gray_rec, _ = model(gray_batch)  # Unpack tuple
                        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
                        
                        out_mask = model_seg(joined_in)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                        # Process mask
                        out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
                        out_mask_averaged = F.avg_pool2d(
                            out_mask_sm[:, 1:, :, :], 21, stride=1, padding=21 // 2
                        ).cpu().numpy()

                        # Store scores
                        image_score = np.max(out_mask_averaged)
                        anomaly_score_prediction.append(image_score)
                        pixel_scores.extend(out_mask_cv.flatten())
                        gt_pixel_scores.extend(true_mask_cv.flatten())

                del gray_batch, gray_rec, joined_in, out_mask, out_mask_sm
                clear_gpu_memory()

            except RuntimeError as e:
                print(f"Error processing batch {i_batch}: {str(e)}")
                continue

        # Convert lists to arrays for metric calculation
        pixel_scores = np.array(pixel_scores)
        gt_pixel_scores = np.array(gt_pixel_scores, dtype=np.uint8)
        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)

        try:
            auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
            ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
            auroc_pixel = roc_auc_score(gt_pixel_scores, pixel_scores)
            ap_pixel = average_precision_score(gt_pixel_scores, pixel_scores)

            # Store results
            obj_ap_pixel_list.append(ap_pixel)
            obj_auroc_pixel_list.append(auroc_pixel)
            obj_auroc_image_list.append(auroc)
            obj_ap_image_list.append(ap)

            # Print results in the exact format requested
            print(f"\nResults for {obj_name}:")
            print(f"AUC Image: {auroc:.3f}")
            print(f"AP Image: {ap:.3f}")
            print(f"AUC Pixel: {auroc_pixel:.3f}")
            print(f"AP Pixel: {ap_pixel:.3f}")

        except Exception as e:
            print(f"Error calculating metrics for {obj_name}: {str(e)}")
            continue

        # Clear models from GPU
        del model, model_seg
        clear_gpu_memory()

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)

    args = parser.parse_args()

    # Set memory-related environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

    obj_list = ['transistor']  # Testing only pill category

    with torch.cuda.device(args.gpu_id):
        test(obj_list, args.data_path, args.checkpoint_path, args.base_model_name)
