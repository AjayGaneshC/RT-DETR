import os
import time
import logging
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from PIL import Image
from pathlib import Path
import glob
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TensorRTInference:
    def __init__(self, engine_path):
        """Initialize TensorRT engine and allocate buffers."""
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Set optimal batch size
        self.batch_size = 1
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.allocate_buffers()
        
        # Warmup
        self.warmup()
    
    def allocate_buffers(self):
        """Allocate device buffers for inputs and outputs."""
        for binding in range(self.engine.num_bindings):
            # Get binding properties
            binding_shape = self.engine.get_binding_shape(binding)
            if binding_shape[0] == -1:  # Dynamic batch size
                binding_shape = (self.batch_size,) + tuple(binding_shape[1:])
                self.context.set_binding_shape(binding, binding_shape)
            
            size = trt.volume(binding_shape)  # The total number of elements in the binding
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            try:
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            except Exception as e:
                logger.error(f"Failed to allocate memory for binding {binding}: {str(e)}")
                logger.info(f"Shape: {binding_shape}, Size: {size}, dtype: {dtype}")
                raise
            
            binding_dict = {
                'host': host_mem,
                'device': device_mem,
                'shape': binding_shape,
                'dtype': dtype,
                'name': self.engine.get_binding_name(binding)
            }
            
            if self.engine.binding_is_input(binding):
                self.inputs.append(binding_dict)
            else:
                self.outputs.append(binding_dict)
            
            logger.info(f"Allocated buffer for {binding_dict['name']}: shape={binding_shape}, size={size}, dtype={dtype}")
    
    def warmup(self):
        """Warmup the engine with dummy data."""
        try:
            if len(self.inputs) > 0:
                input_shape = self.inputs[0]['shape']
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
                self.infer(dummy_input)
                logger.info("Engine warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference."""
        # Read image
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)
        
        # Ensure landscape orientation
        if img_np.shape[0] > img_np.shape[1]:
            img_np = np.rot90(img_np, k=-1)
        
        # Resize to target size
        img_resized = cv2.resize(img_np, (1024, 128))
        
        # Apply Gabor filter (same as training)
        ksize = 31
        sigma = 4.0
        theta = 0
        lambd = 10.0
        gamma = 0.5
        psi = 0
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        img_gabor = cv2.filter2D(img_resized, cv2.CV_8UC3, gabor_kernel)
        
        # Stack and normalize
        img_combined = np.stack((img_resized, img_gabor), axis=0).astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(img_combined, axis=0)
    
    def infer(self, input_data):
        """Run inference on the input data."""
        if not isinstance(input_data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        
        if input_data.shape != self.inputs[0]['shape']:
            raise ValueError(f"Input shape mismatch. Expected {self.inputs[0]['shape']}, got {input_data.shape}")
        
        try:
            # Copy input data to input buffer
            np.copyto(self.inputs[0]['host'], input_data.ravel())
            
            # Transfer input data to device
            cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
            
            # Run inference
            self.context.execute_v2(
                bindings=[inp['device'] for inp in self.inputs] + [out['device'] for out in self.outputs]
            )
            
            # Transfer output data back to host and reshape
            outputs = []
            for out in self.outputs:
                cuda.memcpy_dtoh(out['host'], out['device'])
                output = np.array(out['host'], dtype=out['dtype']).reshape(out['shape'])
                outputs.append(output)
            
            return outputs[0], outputs[1]  # position, confidence
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise

def calculate_iou(pred_box, gt_box):
    """Calculate IoU between predicted and ground truth boxes."""
    pred_left = pred_box[0] - pred_box[1] / 2
    pred_right = pred_box[0] + pred_box[1] / 2
    gt_left = gt_box[0] - gt_box[1] / 2
    gt_right = gt_box[0] + gt_box[1] / 2
    
    intersection = max(0, min(pred_right, gt_right) - max(pred_left, gt_left))
    union = (pred_right - pred_left) + (gt_right - gt_left) - intersection
    
    return intersection / (union + 1e-6)

def read_label_file(label_path):
    """Read ground truth from label file."""
    if not os.path.exists(label_path):
        return None, 0.0
    
    try:
        with open(label_path, 'r') as f:
            content = f.read().strip()
            if not content:
                return None, 0.0
            
            values = list(map(float, content.split()))
            if len(values) == 5:  # class_id, x_center, y_center, width, confidence
                x_center = values[1] * 1024  # Denormalize
                width = values[3] * 1024     # Denormalize
                return np.array([x_center, width]), 1.0
    except Exception as e:
        logger.warning(f"Error reading label file {label_path}: {str(e)}")
    
    return None, 0.0

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Engine path (converted from ONNX using trtexec)
    engine_path = os.path.join(current_dir, "model_fp16.engine")
    
    if not os.path.exists(engine_path):
        logger.error(f"Engine file not found: {engine_path}")
        logger.info("Please run trtexec to generate the engine file first.")
        return
    
    try:
        # Initialize inference engine
        logger.info(f"Initializing TensorRT engine from {engine_path}...")
        engine = TensorRTInference(engine_path)
        
        # Test dataset paths - using absolute paths
        test_img_dir = os.path.join(current_dir, "data/images/test")
        test_label_dir = os.path.join(current_dir, "data/labels/test")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(current_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(test_img_dir):
            logger.error(f"Test image directory not found: {test_img_dir}")
            # Try to list contents of data directory to help debugging
            data_dir = os.path.join(current_dir, "data")
            if os.path.exists(data_dir):
                logger.info("Available directories in data/:")
                for item in os.listdir(data_dir):
                    logger.info(f"  - {item}")
            return
            
        # Get all test images
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(glob.glob(os.path.join(test_img_dir, ext)))
        test_images = sorted(test_images)
        
        if not test_images:
            logger.error(f"No test images found in {test_img_dir}!")
            # List contents of the directory
            if os.path.exists(test_img_dir):
                logger.info("Contents of test image directory:")
                for item in os.listdir(test_img_dir):
                    logger.info(f"  - {item}")
            return
        
        # Metrics
        total_images = len(test_images)
        total_time = 0
        results = []
        
        logger.info(f"Found {total_images} test images")
        logger.info(f"Running inference on {total_images} test images...")
        
        for img_path in tqdm(test_images):
            try:
                # Log the image being processed
                logger.debug(f"Processing image: {img_path}")
                
                # Preprocess and run inference
                input_data = engine.preprocess_image(img_path)
                start_time = time.time()
                position, confidence = engine.infer(input_data)
                inference_time = (time.time() - start_time) * 1000
                
                # Store results
                results.append({
                    'image': os.path.basename(img_path),
                    'pred_center': float(position[0, 0]),
                    'pred_width': float(position[0, 1]),
                    'pred_conf': float(confidence[0]),
                    'inference_time': inference_time
                })
                
                total_time += inference_time
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Calculate and display results
        if results:
            avg_inference_time = total_time / len(results)
            logger.info(f"\nResults Summary:")
            logger.info(f"Average inference time: {avg_inference_time:.2f}ms")
            logger.info(f"Processed {len(results)} images successfully")
            
            # Save detailed results
            results_file = os.path.join(output_dir, "inference_results.txt")
            with open(results_file, 'w') as f:
                f.write("Inference Results\n")
                f.write("================\n\n")
                f.write(f"Test directory: {test_img_dir}\n")
                f.write(f"Engine file: {engine_path}\n")
                f.write(f"Total images processed: {len(results)}\n")
                f.write(f"Average inference time: {avg_inference_time:.2f}ms\n\n")
                f.write("Detailed Results:\n")
                for r in results:
                    f.write(f"\nImage: {r['image']}\n")
                    f.write(f"  Center: {r['pred_center']:.1f}\n")
                    f.write(f"  Width: {r['pred_width']:.1f}\n")
                    f.write(f"  Confidence: {r['pred_conf']:.3f}\n")
                    f.write(f"  Inference Time: {r['inference_time']:.2f}ms\n")
            
            logger.info(f"Detailed results saved to {results_file}")
        else:
            logger.error("No results were generated!")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
