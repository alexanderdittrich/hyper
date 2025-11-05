import os
import wandb
import numpy as np
from PIL import Image


class WandbLogger:
    """
    Logger for Weights & Biases integration
    """

    def __init__(self, args, exp_label):
        self.args = args
        self.exp_label = exp_label

        # Create output folder
        results_log_dir = (
            args.results_log_dir if args.results_log_dir is not None else "./logs"
        )
        self.full_output_folder = os.path.join(results_log_dir, exp_label)
        if not os.path.exists(self.full_output_folder):
            os.makedirs(self.full_output_folder)

        # Initialize wandb
        wandb.init(
            project=getattr(args, "wandb_project", "hyper"),
            name=exp_label,
            config=vars(args),
            dir=self.full_output_folder,
            save_code=True,
            mode=getattr(args, "wandb_mode", "online"),
        )

        # Define custom x-axes for metrics
        # Metrics with "_per_frame" will use frames as x-axis
        # Metrics with "_per_iter" will use iterations as x-axis
        wandb.define_metric("frames")
        wandb.define_metric("iterations")
        wandb.define_metric("return_avg_per_frame/*", step_metric="frames")
        wandb.define_metric("return_std_per_frame/*", step_metric="frames")
        wandb.define_metric("return_avg_per_iter/*", step_metric="iterations")
        wandb.define_metric("return_std_per_iter/*", step_metric="iterations")
        wandb.define_metric("Meta-Episode Return*", step_metric="frames")
        wandb.define_metric("Meta-Episode Success*", step_metric="frames")
        wandb.define_metric("Grad_Norm*", step_metric="frames")
        wandb.define_metric("State_Grad_Norm*", step_metric="frames")
        wandb.define_metric("policy_losses/*", step_metric="iterations")
        wandb.define_metric("policy/*", step_metric="iterations")
        wandb.define_metric("vae_losses/*", step_metric="iterations")
        wandb.define_metric("encoder/*", step_metric="iterations")
        wandb.define_metric("decoder/*", step_metric="iterations")
        wandb.define_metric("weights/*", step_metric="iterations")
        wandb.define_metric("gradients/*", step_metric="iterations")

        # Store wandb run
        self.run = wandb.run

        print(f"🔗 Logging to wandb: {wandb.run.url}")
        print(f"📁 Local logs: {self.full_output_folder}")

    def add(self, name, value, step):
        """
        Log a scalar value

        Args:
            name: metric name (can include '/' for grouping)
            value: scalar value or torch tensor
            step: step number (iteration or frame count)
        """
        # Convert torch tensors to numpy/python
        if hasattr(value, "item"):
            value = value.item()
        elif hasattr(value, "cpu"):
            value = value.cpu().numpy()

        # Determine which step metric to update based on metric name
        log_dict = {name: value}
        if "_per_frame" in name or "Meta-Episode" in name:
            log_dict["frames"] = step
        elif "_per_iter" in name or any(
            x in name
            for x in ["policy", "vae", "encoder", "decoder", "weights", "gradients"]
        ):
            log_dict["iterations"] = step
        else:
            # Default: use both for compatibility
            log_dict["frames"] = step
            log_dict["iterations"] = step

        wandb.log(log_dict, commit=True)

    def add_image(self, name, image, step):
        """
        Log an image

        Args:
            name: image name
            image: numpy array (H, W, C) or PIL Image
            step: step number
        """
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image if needed
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        wandb.log({name: wandb.Image(image)}, step=step)

    def add_video(self, name, video, step, fps=30):
        """
        Log a video

        Args:
            name: video name
            video: numpy array (T, H, W, C) or (T, C, H, W)
            step: step number
            fps: frames per second
        """
        # Ensure video is in (T, H, W, C) format
        if video.shape[1] == 3 or video.shape[1] == 1:  # (T, C, H, W) -> (T, H, W, C)
            video = np.transpose(video, (0, 2, 3, 1))

        wandb.log({name: wandb.Video(video, fps=fps, format="mp4")}, step=step)

    def add_histogram(self, name, values, step):
        """
        Log a histogram

        Args:
            name: histogram name
            values: array of values
            step: step number
        """
        if hasattr(values, "cpu"):
            values = values.cpu().numpy()

        wandb.log({name: wandb.Histogram(values)}, step=step)

    def finish(self):
        """
        Finish the wandb run
        """
        wandb.finish()
