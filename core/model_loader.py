"""
MORPHEUS CHAT - Advanced Model Loading and Management System
Production-grade model orchestration with hot-swapping, quantization, and multi-modal support.

Architecture:
- Unified interface for local and API-based models
- Dynamic quantization with performance optimization
- Memory-efficient loading with gradient checkpointing
- Hardware acceleration detection and utilization
- Model registry with capability-based routing
- Hot-swapping with minimal service interruption
"""

import asyncio
import gc
import logging
import psutil
import time
import torch
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig, GenerationConfig,
    pipeline, Pipeline
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from optimum.bettertransformer import BetterTransformer
import accelerate
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from schemas.models import (
    ModelConfiguration, ModelRegistry, ModelLoadRequest, ModelLoadResponse,
    ModelType, QuantizationMethod, ModelCapability, ModelID
)

logger = logging.getLogger(__name__)


class HardwareManager:
    """
    Advanced hardware detection and optimization for model loading.
    Implements intelligent device allocation and performance tuning.
    """
    
    def __init__(self):
        self.available_devices = self._detect_devices()
        self.device_capabilities = self._analyze_device_capabilities()
        self.memory_info = self._get_memory_info()
        
        logger.info(f"Hardware manager initialized: {len(self.available_devices)} devices detected")
    
    def _detect_devices(self) -> List[str]:
        """Detect available hardware acceleration devices"""
        devices = ["cpu"]
        
        # CUDA detection
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
            logger.info(f"CUDA devices detected: {torch.cuda.device_count()}")
        
        # MPS detection (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
            logger.info("MPS (Apple Silicon) device detected")
        
        # Intel XPU detection (future support)
        try:
            import intel_extension_for_pytorch as ipex
            if ipex.xpu.is_available():
                devices.append("xpu")
                logger.info("Intel XPU device detected")
        except ImportError:
            pass
        
        return devices
    
    def _analyze_device_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Analyze detailed capabilities of each device"""
        capabilities = {}
        
        for device in self.available_devices:
            if device == "cpu":
                capabilities[device] = {
                    "type": "cpu",
                    "memory_gb": psutil.virtual_memory().total / (1024**3),
                    "cores": psutil.cpu_count(logical=True),
                    "cores_physical": psutil.cpu_count(logical=False),
                    "supports_fp16": False,
                    "supports_bf16": False,
                    "compute_capability": None
                }
            
            elif device.startswith("cuda"):
                gpu_id = int(device.split(":")[1])
                props = torch.cuda.get_device_properties(gpu_id)
                
                capabilities[device] = {
                    "type": "cuda",
                    "name": props.name,
                    "memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "supports_fp16": props.major >= 5,  # Maxwell and newer
                    "supports_bf16": props.major >= 8,  # Ampere and newer
                    "multiprocessors": props.multi_processor_count,
                    "max_threads_per_block": props.max_threads_per_block
                }
            
            elif device == "mps":
                capabilities[device] = {
                    "type": "mps",
                    "memory_gb": psutil.virtual_memory().total / (1024**3),  # Unified memory
                    "supports_fp16": True,
                    "supports_bf16": False,  # Limited support
                    "compute_capability": "apple_silicon"
                }
        
        return capabilities
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        memory_info = {
            "system_total_gb": psutil.virtual_memory().total / (1024**3),
            "system_available_gb": psutil.virtual_memory().available / (1024**3),
            "system_used_percent": psutil.virtual_memory().percent
        }
        
        # GPU memory info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                
                memory_info[f"cuda_{i}_total_gb"] = total
                memory_info[f"cuda_{i}_allocated_gb"] = allocated
                memory_info[f"cuda_{i}_cached_gb"] = cached
                memory_info[f"cuda_{i}_free_gb"] = total - cached
        
        return memory_info
    
    def select_optimal_device(self, memory_requirement_gb: float, preferred_device: Optional[str] = None) -> str:
        """
        Select optimal device for model loading based on requirements and availability.
        Implements intelligent device selection with performance optimization.
        """
        if preferred_device and preferred_device in self.available_devices:
            device_caps = self.device_capabilities[preferred_device]
            if device_caps.get("memory_gb", 0) >= memory_requirement_gb:
                return preferred_device
            else:
                logger.warning(f"Preferred device {preferred_device} has insufficient memory")
        
        # Score devices based on capabilities and availability
        device_scores = []
        
        for device in self.available_devices:
            caps = self.device_capabilities[device]
            available_memory = caps.get("memory_gb", 0)
            
            # Skip devices with insufficient memory
            if available_memory < memory_requirement_gb:
                continue
            
            score = 0
            
            # Prefer GPU over CPU
            if caps["type"] == "cuda":
                score += 100
                # Prefer newer compute capabilities
                if caps.get("compute_capability"):
                    major, minor = map(int, caps["compute_capability"].split("."))
                    score += major * 10 + minor
            elif caps["type"] == "mps":
                score += 80
            elif caps["type"] == "cpu":
                score += 10
            
            # Memory bonus (more available memory is better)
            memory_ratio = available_memory / memory_requirement_gb
            score += min(memory_ratio * 10, 50)
            
            # FP16/BF16 support bonus
            if caps.get("supports_fp16"):
                score += 5
            if caps.get("supports_bf16"):
                score += 10
            
            device_scores.append((device, score))
        
        if not device_scores:
            logger.warning(f"No devices found with sufficient memory ({memory_requirement_gb:.1f}GB)")
            return "cpu"  # Fallback to CPU
        
        # Select device with highest score
        best_device = max(device_scores, key=lambda x: x[1])[0]
        logger.info(f"Selected device {best_device} for model loading")
        return best_device
    
    def get_device_utilization(self) -> Dict[str, float]:
        """Get current device utilization metrics"""
        utilization = {}
        
        # CPU utilization
        utilization["cpu_percent"] = psutil.cpu_percent(interval=1)
        
        # GPU utilization
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # Memory utilization
                    total_mem = torch.cuda.get_device_properties(i).total_memory
                    used_mem = torch.cuda.memory_allocated(i)
                    utilization[f"cuda_{i}_memory_percent"] = (used_mem / total_mem) * 100
                    
                    # GPU utilization (requires nvidia-ml-py or similar)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        utilization[f"cuda_{i}_gpu_percent"] = gpu_util.gpu
                    except ImportError:
                        pass
                except Exception as e:
                    logger.warning(f"Failed to get GPU {i} utilization: {e}")
        
        return utilization


class QuantizationManager:
    """
    Advanced quantization management with dynamic optimization.
    Implements ChatGPT-equivalent model compression with quality preservation.
    """
    
    def __init__(self):
        self.supported_methods = self._detect_quantization_support()
        logger.info(f"Quantization manager initialized: {list(self.supported_methods.keys())}")
    
    def _detect_quantization_support(self) -> Dict[str, bool]:
        """Detect available quantization methods"""
        support = {}
        
        # BitsAndBytes (CUDA only)
        try:
            import bitsandbytes as bnb
            support["bitsandbytes"] = torch.cuda.is_available()
        except ImportError:
            support["bitsandbytes"] = False
        
        # GPTQ
        try:
            from auto_gptq import AutoGPTQForCausalLM
            support["gptq"] = True
        except ImportError:
            support["gptq"] = False
        
        # AWQ
        try:
            from awq import AutoAWQForCausalLM
            support["awq"] = True
        except ImportError:
            support["awq"] = False
        
        # PyTorch native quantization
        support["pytorch_dynamic"] = True
        support["pytorch_static"] = True
        
        return support
    
    def create_quantization_config(self, config: Dict[str, Any], device: str) -> Optional[Any]:
        """
        Create quantization configuration based on method and device.
        Returns appropriate quantization config object.
        """
        if not config.get("enabled", False):
            return None
        
        method = config.get("method", QuantizationMethod.BITSANDBYTES)
        bits = config.get("bits", 4)
        
        if method == QuantizationMethod.BITSANDBYTES and self.supported_methods.get("bitsandbytes"):
            if not device.startswith("cuda"):
                logger.warning("BitsAndBytes requires CUDA, falling back to no quantization")
                return None
            
            from transformers import BitsAndBytesConfig
            
            if bits == 4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=config.get("double_quantization", True),
                    bnb_4bit_quant_type=config.get("quant_type", "nf4")
                )
            elif bits == 8:
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_skip_modules=config.get("skip_modules", [])
                )
        
        elif method == QuantizationMethod.GPTQ and self.supported_methods.get("gptq"):
            # GPTQ configuration would be handled during model loading
            return {
                "method": "gptq",
                "bits": bits,
                "group_size": config.get("group_size", 128),
                "desc_act": config.get("desc_act", False)
            }
        
        elif method == QuantizationMethod.AWQ and self.supported_methods.get("awq"):
            # AWQ configuration
            return {
                "method": "awq",
                "bits": bits,
                "group_size": config.get("group_size", 128)
            }
        
        elif method in [QuantizationMethod.DYNAMIC_INT8, QuantizationMethod.STATIC_INT8]:
            # PyTorch native quantization
            return {
                "method": method.value,
                "bits": 8,
                "backend": config.get("backend", "fbgemm" if device == "cpu" else "qnnpack")
            }
        
        logger.warning(f"Quantization method {method} not supported on device {device}")
        return None
    
    def apply_quantization(self, model: torch.nn.Module, config: Dict[str, Any], device: str) -> torch.nn.Module:
        """
        Apply post-loading quantization if needed.
        Used for methods that require model-level quantization.
        """
        if not config or config.get("method") not in ["pytorch_dynamic", "pytorch_static"]:
            return model
        
        method = config["method"]
        
        if method == "pytorch_dynamic":
            # Dynamic quantization (CPU only)
            if device == "cpu":
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("Applied PyTorch dynamic quantization")
                return quantized_model
        
        elif method == "pytorch_static":
            # Static quantization requires calibration dataset
            logger.warning("Static quantization not yet implemented")
        
        return model


class ModelLoader:
    """
    Production-grade model loader with advanced optimization and error recovery.
    Implements ChatGPT-equivalent model management with hot-swapping capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hardware_manager = HardwareManager()
        self.quantization_manager = QuantizationManager()
        
        # Model registry and cache
        self.model_registry = ModelRegistry()
        self.loaded_models: Dict[ModelID, Dict[str, Any]] = {}
        self.model_cache: Dict[str, Any] = {}
        self.load_lock = threading.RLock()
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="model-loader")
        
        # Load model registry from config
        self._initialize_model_registry()
        
        logger.info(f"Model loader initialized with {len(self.model_registry.models)} registered models")
    
    def _initialize_model_registry(self):
        """Initialize model registry from configuration"""
        models_config = self.config.get("model_registry", {})
        
        # Load local models
        local_models = models_config.get("local_models", {})
        for model_id, model_config in local_models.items():
            try:
                config_obj = ModelConfiguration(
                    model_id=model_id,
                    **model_config
                )
                self.model_registry.register_model(config_obj)
                logger.info(f"Registered local model: {model_id}")
            except Exception as e:
                logger.error(f"Failed to register model {model_id}: {e}")
        
        # Load API models
        api_models = models_config.get("api_models", {})
        for model_id, model_config in api_models.items():
            try:
                config_obj = ModelConfiguration(
                    model_id=model_id,
                    model_type=ModelType.API_PROXY,
                    **model_config
                )
                self.model_registry.register_model(config_obj)
                logger.info(f"Registered API model: {model_id}")
            except Exception as e:
                logger.error(f"Failed to register API model {model_id}: {e}")
    
    async def load_model(self, request: ModelLoadRequest) -> ModelLoadResponse:
        """
        Load model with comprehensive optimization and error handling.
        Implements intelligent device selection and memory management.
        """
        model_id = request.model_id
        start_time = time.time()
        
        logger.info(f"Loading model: {model_id}")
        
        try:
            # Check if model is already loaded
            with self.load_lock:
                if model_id in self.loaded_models:
                    existing_model = self.loaded_models[model_id]
                    logger.info(f"Model {model_id} already loaded")
                    
                    return ModelLoadResponse(
                        model_id=model_id,
                        success=True,
                        message="Model already loaded",
                        load_time_seconds=0.0,
                        memory_usage_gb=existing_model.get("memory_usage_gb", 0),
                        model_info=existing_model.get("info", {})
                    )
            
            # Get model configuration
            model_config = self.model_registry.models.get(model_id)
            if not model_config:
                return ModelLoadResponse(
                    model_id=model_id,
                    success=False,
                    message=f"Model {model_id} not found in registry",
                    load_time_seconds=0.0,
                    memory_usage_gb=0.0,
                    error_type="model_not_found"
                )
            
            # Handle API models differently
            if model_config.model_type == ModelType.API_PROXY:
                return await self._load_api_model(model_config, request, start_time)
            
            # Load local model
            return await self._load_local_model(model_config, request, start_time)
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Failed to load model {model_id}: {e}")
            
            return ModelLoadResponse(
                model_id=model_id,
                success=False,
                message=f"Model loading failed: {str(e)}",
                load_time_seconds=load_time,
                memory_usage_gb=0.0,
                error_type="loading_error",
                error_details=str(e)
            )
    
    async def _load_local_model(self, model_config: ModelConfiguration, request: ModelLoadRequest, start_time: float) -> ModelLoadResponse:
        """Load local transformer model with optimizations"""
        model_id = model_config.model_id
        
        # Device selection
        memory_requirement = model_config.resource_requirements.gpu_memory_gb
        device = self.hardware_manager.select_optimal_device(
            memory_requirement, 
            request.device
        )
        
        # Quantization configuration
        quantization_config = None
        if model_config.quantization.enabled:
            quantization_config = self.quantization_manager.create_quantization_config(
                model_config.quantization.dict(),
                device
            )
        
        # Override with request parameters
        if request.quantization_override:
            quantization_config = self.quantization_manager.create_quantization_config(
                request.quantization_override.dict(),
                device
            )
        
        try:
            # Load model in executor to avoid blocking
            model_data = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._load_model_sync,
                model_config,
                device,
                quantization_config
            )
            
            load_time = time.time() - start_time
            
            # Store in loaded models
            with self.load_lock:
                self.loaded_models[model_id] = {
                    "model": model_data["model"],
                    "tokenizer": model_data["tokenizer"],
                    "pipeline": model_data.get("pipeline"),
                    "device": device,
                    "memory_usage_gb": model_data["memory_usage_gb"],
                    "config": model_config,
                    "loaded_at": datetime.now(timezone.utc),
                    "info": model_data["info"]
                }
            
            logger.info(f"Model {model_id} loaded successfully in {load_time:.2f}s")
            
            return ModelLoadResponse(
                model_id=model_id,
                success=True,
                message="Model loaded successfully",
                load_time_seconds=load_time,
                memory_usage_gb=model_data["memory_usage_gb"],
                estimated_tokens_per_second=model_data.get("tokens_per_second"),
                model_info=model_data["info"],
                device_allocation={device: "primary"}
            )
            
        except Exception as e:
            load_time = time.time() - start_time
            raise RuntimeError(f"Local model loading failed: {e}")
    
    def _load_model_sync(self, model_config: ModelConfiguration, device: str, quantization_config: Any) -> Dict[str, Any]:
        """
        Synchronous model loading with comprehensive optimization.
        Runs in thread executor to avoid blocking async loop.
        """
        model_path = str(model_config.model_path)
        
        # Memory tracking
        initial_memory = self._get_memory_usage()
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,  # Security
            use_fast=True
        )
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model configuration
        model_arch_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=False
        )
        
        # Configure model loading parameters
        model_kwargs = {
            "config": model_arch_config,
            "torch_dtype": torch.float16 if device.startswith("cuda") else torch.float32,
            "device_map": "auto" if device.startswith("cuda") else None,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }
        
        # Add quantization config if available
        if quantization_config and hasattr(quantization_config, "load_in_4bit"):
            model_kwargs["quantization_config"] = quantization_config
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if device != "cpu" and not model_kwargs.get("device_map"):
            model = model.to(device)
        
        # Apply post-loading optimizations
        model = self._optimize_model(model, model_config, device)
        
        # Apply post-loading quantization if needed
        if quantization_config and not hasattr(quantization_config, "load_in_4bit"):
            model = self.quantization_manager.apply_quantization(model, quantization_config, device)
        
        # Create pipeline for easier inference
        pipeline_obj = None
        try:
            pipeline_obj = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device.startswith("cuda") else -1,
                torch_dtype=model.dtype,
                trust_remote_code=False
            )
        except Exception as e:
            logger.warning(f"Failed to create pipeline: {e}")
        
        # Calculate memory usage
        final_memory = self._get_memory_usage()
        memory_usage_gb = (final_memory - initial_memory) / (1024**3)
        
        # Estimate performance
        tokens_per_second = self._estimate_performance(model, tokenizer, device)
        
        # Model info
        model_info = {
            "architecture": model_arch_config.architectures[0] if model_arch_config.architectures else "unknown",
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "vocab_size": len(tokenizer),
            "max_position_embeddings": getattr(model_arch_config, "max_position_embeddings", "unknown"),
            "dtype": str(model.dtype),
            "device": str(device),
            "quantized": quantization_config is not None,
            "memory_usage_gb": memory_usage_gb
        }
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "pipeline": pipeline_obj,
            "memory_usage_gb": memory_usage_gb,
            "tokens_per_second": tokens_per_second,
            "info": model_info
        }
    
    def _optimize_model(self, model: torch.nn.Module, config: ModelConfiguration, device: str) -> torch.nn.Module:
        """Apply model optimizations based on configuration"""
        optimizations = config.optimizations
        
        # Torch compile (PyTorch 2.0+)
        if optimizations.torch_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(
                    model,
                    mode=optimizations.torch_compile_mode,
                    backend=optimizations.torch_compile_backend
                )
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # BetterTransformer optimization
        try:
            if device.startswith("cuda"):  # BetterTransformer works best on GPU
                model = BetterTransformer.transform(model)
                logger.info("Applied BetterTransformer optimization")
        except Exception as e:
            logger.warning(f"BetterTransformer optimization failed: {e}")
        
        # Gradient checkpointing (for memory efficiency during inference)
        if optimizations.optimizations.get("gradient_checkpointing"):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
        
        return model
    
    async def _load_api_model(self, model_config: ModelConfiguration, request: ModelLoadRequest, start_time: float) -> ModelLoadResponse:
        """Load API-based model (proxy setup)"""
        model_id = model_config.model_id
        
        try:
            # Validate API configuration
            api_config = model_config.api_config
            if not api_config:
                raise ValueError("API configuration missing")
            
            # Test API connectivity (basic health check)
            # This would be implemented based on the specific API provider
            
            load_time = time.time() - start_time
            
            # Store API model info
            with self.load_lock:
                self.loaded_models[model_id] = {
                    "model": None,  # No local model
                    "tokenizer": None,  # API handles tokenization
                    "api_config": api_config,
                    "device": "api",
                    "memory_usage_gb": 0.0,  # No local memory usage
                    "config": model_config,
                    "loaded_at": datetime.now(timezone.utc),
                    "info": {
                        "provider": api_config.provider,
                        "model_id": api_config.model_id,
                        "type": "api_proxy"
                    }
                }
            
            logger.info(f"API model {model_id} configured successfully")
            
            return ModelLoadResponse(
                model_id=model_id,
                success=True,
                message="API model configured successfully",
                load_time_seconds=load_time,
                memory_usage_gb=0.0,
                model_info={"provider": api_config.provider, "type": "api_proxy"}
            )
            
        except Exception as e:
            load_time = time.time() - start_time
            raise RuntimeError(f"API model configuration failed: {e}")
    
    async def unload_model(self, model_id: ModelID) -> bool:
        """
        Unload model and free resources.
        Implements graceful cleanup with memory optimization.
        """
        try:
            with self.load_lock:
                if model_id not in self.loaded_models:
                    logger.warning(f"Model {model_id} not loaded")
                    return False
                
                model_data = self.loaded_models.pop(model_id)
            
            # Clean up model resources
            if model_data.get("model"):
                del model_data["model"]
            if model_data.get("tokenizer"):
                del model_data["tokenizer"]
            if model_data.get("pipeline"):
                del model_data["pipeline"]
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model {model_id} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    async def switch_model(self, current_model_id: ModelID, new_model_id: ModelID) -> ModelLoadResponse:
        """
        Hot-swap models with minimal service interruption.
        Implements intelligent preloading and graceful transition.
        """
        logger.info(f"Switching from {current_model_id} to {new_model_id}")
        
        try:
            # Load new model
            load_request = ModelLoadRequest(model_id=new_model_id)
            load_response = await self.load_model(load_request)
            
            if not load_response.success:
                return load_response
            
            # Unload old model (optional - could keep for fallback)
            if current_model_id != new_model_id:
                await self.unload_model(current_model_id)
            
            logger.info(f"Model switch completed: {current_model_id} -> {new_model_id}")
            return load_response
            
        except Exception as e:
            logger.error(f"Model switch failed: {e}")
            return ModelLoadResponse(
                model_id=new_model_id,
                success=False,
                message=f"Model switch failed: {str(e)}",
                load_time_seconds=0.0,
                memory_usage_gb=0.0,
                error_type="switch_error"
            )
    
    def get_loaded_model(self, model_id: ModelID) -> Optional[Dict[str, Any]]:
        """Get loaded model data"""
        with self.load_lock:
            return self.loaded_models.get(model_id)
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """List all loaded models with metadata"""
        with self.load_lock:
            return [
                {
                    "model_id": model_id,
                    "device": data["device"],
                    "memory_usage_gb": data["memory_usage_gb"],
                    "loaded_at": data["loaded_at"].isoformat(),
                    "info": data["info"]
                }
                for model_id, data in self.loaded_models.items()
            ]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            return psutil.Process().memory_info().rss
    
    def _estimate_performance(self, model: torch.nn.Module, tokenizer, device: str) -> Optional[float]:
        """Estimate model performance (tokens per second)"""
        try:
            # Simple performance test
            test_input = "Hello, how are you today?"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Warm up
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            # Timing test
            start_time = time.time()
            num_tokens = 50
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=num_tokens, 
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            elapsed = time.time() - start_time
            tokens_per_second = num_tokens / elapsed
            
            return tokens_per_second
            
        except Exception as e:
            logger.warning(f"Performance estimation failed: {e}")
            return None
    
    def get_model_registry(self) -> ModelRegistry:
        """Get the model registry"""
        return self.model_registry
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "hardware": {
                "devices": self.hardware_manager.available_devices,
                "device_capabilities": self.hardware_manager.device_capabilities,
                "memory_info": self.hardware_manager.get_memory_info(),
                "utilization": self.hardware_manager.get_device_utilization()
            },
            "quantization": {
                "supported_methods": self.quantization_manager.supported_methods
            },
            "loaded_models": len(self.loaded_models),
            "registered_models": len(self.model_registry.models),
            "total_memory_usage_gb": sum(
                data["memory_usage_gb"] for data in self.loaded_models.values()
            )
        }