import os
import numpy as np
import torch
import torchvision
import json
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import prefilter_voxel, render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, logger, split_idx, stage, add_dist):
    warmup_iter = 4999
    if args.stage == "uw_pretrain":
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist = add_dist, add_dist, add_dist
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                                  dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                                  dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist,
                                  dataset.add_color_dist)
        scene = Scene(dataset, gaussians, shuffle=False)
        gaussians.training_setup(opt)

        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_stack_far = viewpoint_stack[:split_idx] # 10
        print(f"num of far = {len(viewpoint_stack_far)}")
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, warmup_iter), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, warmup_iter + 1):
            iter_start.record()
            gaussians.update_learning_rate(iteration)
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            # Pick a random Camera
            viewpoint_cam_far = viewpoint_stack_far[randint(0, len(viewpoint_stack_far) - 1)]

            if (iteration - 1) == debug_from:
                pipe.debug = True
            # far
            voxel_visible_mask = prefilter_voxel(viewpoint_cam_far, gaussians, pipe, background)
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            render_pkg = render(viewpoint_cam_far, gaussians, pipe, background, visible_mask=voxel_visible_mask,
                                retain_grad=retain_grad, grow=False)
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg[
                "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], \
            render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
            gt_image = viewpoint_cam_far.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = (1.0 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()
            loss= (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
            loss.backward()

            iter_end.record()

            with torch.no_grad():# Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == warmup_iter:
                    progress_bar.close()
                    logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                # Log and save
                if iteration > opt.start_stat:# densification
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask,
                                              voxel_visible_mask)
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold,
                                                grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                if iteration < opt.iterations:# Optimizer step
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

    elif args.stage == "uw2wide":
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist = add_dist, add_dist, add_dist
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                                  dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                                  dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist,
                                  dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=warmup_iter, shuffle=False)
        gaussians.training_setup(opt)
        scene.gaussians.train() # need to add
        
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_stack_far = viewpoint_stack[:split_idx] # 10
        viewpoint_stack_near = viewpoint_stack[split_idx:]
        print(f"num of far:near = {len(viewpoint_stack_far)}:{len(viewpoint_stack_near)}")
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, opt.iterations + 1):
            iter_start.record()
            gaussians.update_learning_rate(iteration)
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            # Pick a random Camera
            viewpoint_cam_far = viewpoint_stack_far[randint(0, len(viewpoint_stack_far) - 1)]
            viewpoint_cam_near = viewpoint_stack_near[randint(0, len(viewpoint_stack_near) - 1)]

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            # far
            voxel_visible_mask = prefilter_voxel(viewpoint_cam_far, gaussians, pipe, background)
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            render_pkg = render(viewpoint_cam_far, gaussians, pipe, background, visible_mask=voxel_visible_mask,
                                retain_grad=retain_grad, offset=False)
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg[
                "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], \
            render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
            gt_image = viewpoint_cam_far.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = (1.0 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()
            loss_far = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
            # near
            # generate_neural_gaussian-->gradients-->masked_gaussians-->MLP-->gaussian_growing
            voxel_visible_mask = prefilter_voxel(viewpoint_cam_near, gaussians, pipe, background)
            render_pkg = render(viewpoint_cam_near, gaussians, pipe, background, visible_mask=voxel_visible_mask,
                                retain_grad=retain_grad, offset=True)
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg[
                "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], \
            render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
            gt_image = viewpoint_cam_near.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = (1.0 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()
            loss_near = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg

            loss = loss_far + loss_near
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                # Log and save
                if (iteration in saving_iterations):
                    logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                # densification
                if iteration < opt.update_until and iteration > opt.start_stat:
                    # add statis
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask,
                                              voxel_visible_mask)
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold,
                                                grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                elif iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()
                if iteration < opt.iterations:# Optimizer step
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                if (iteration in checkpoint_iterations):
                    logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def get_logger(path):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--split_idx", type=int, default=10) # far[:10], near[10:]
    parser.add_argument("--stage", type=str, default=None, help="uw_pretrain, uw2wide")
    parser.add_argument("--add_dist", action="store_true", help="add dist into scaffold")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # enable logging
    os.makedirs(args.model_path, exist_ok=True)
    logger = get_logger(args.model_path)
    logger.info(f'args: {args}')
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, logger=logger, split_idx=args.split_idx, stage=args.stage, add_dist=args.add_dist)

    # All done
    logger.info("\nTraining complete.")
