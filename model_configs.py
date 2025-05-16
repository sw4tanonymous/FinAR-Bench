"""
model_configs.py - define all supported models and their configuration parameters

This module contains three types of model configurations:
1. OpenAI interface models (including NVIDIA and native OpenAI)
2. Volcano engine interface models
3. Custom interface models

How to add new models:
----------------------

1. Add OpenAI interface models:
   Add a new key-value pair in the OPENAI_INTERFACE_MODELS dictionary, for example:
   ```
   "new-model-name": {
       "temperature": 0.6,
       "top_p": 0.7,
       "max_tokens": 4000,
       "stream": False,
       "base_url": "https://api.example.com/v1",
       "api_key": "your-api-key",  
       "provider": "nvidia"  # or "openai" and other providers
   }
   ```

2. Add Volcano engine interface models:
   Add a new key-value pair in the VOLCENGINE_INTERFACE_MODELS dictionary, for example:
   ```
   "volcengine-new-model": {
       "temperature": 0.6,
       "top_p": 0.7,
       "max_tokens": 4000,
       "stream": False,
       "model_id": "your-model-id",  # the model ID of Volcano engine
       "api_key": "your-api-key",    # recommend to use environment variable
       "provider": "volcengine"
   }
   ```

3. Add custom interface models:
   Add a new key-value pair in the CUSTOM_INTERFACE_MODELS dictionary.
   
   a. Use the existing OpenAI client:
   ```
   "my-custom-openai-model": {
       "temperature": 0.7,
       "top_p": 0.8,
       "max_tokens": 4000,
       "base_url": "https://api.example.com/v1",
       "api_key": "your-api-key",
       "provider": "custom",
       "client_type": "openai"  # use OpenAI client
   }
   ```
   
   b. Use the existing Volcano engine client:
   ```
   "my-custom-volcengine-model": {
       "temperature": 0.7,
       "top_p": 0.8,
       "max_tokens": 4000,
       "api_key": "your-api-key",
       "provider": "custom",
       "client_type": "volcengine"  # use Volcano engine client
   }
   ```
   
   c. Use a fully custom client:
   ```
   "my-fully-custom-model": {
       "temperature": 0.7,
       "top_p": 0.8,
       "max_tokens": 4000,
       "base_url": "https://api.example.com/v1",
       "api_key": "your-api-key",
       "provider": "custom",
       "client_type": "custom_module",
       "custom_module": "my_package.my_client",  # the path of the custom module
       "custom_class": "MyAPIClient",            # the name of the custom class
       "extra_args": {                           # extra initialization parameters
           "timeout": 600,
           "other_param": "value"
       }
   }
   ```
   Note: The custom client class must accept the api_key and base_url parameters, and provide interfaces compatible with the OpenAI client.

Environment variables:
----------------------
The system will automatically find the API keys in the following environment variables:
- NVIDIA_API_KEY: for NVIDIA models
- OPENAI_API_KEY: for OpenAI models
- VOLCENGINE_API_KEY: for Volcano engine models

If the API key is not specified in the configuration, the system will use these environment variables.
"""

INTERFACE_TYPE_OPENAI = "openai"  # the API using OpenAI interface format (including NVIDIA and native OpenAI)
INTERFACE_TYPE_VOLCENGINE = "volcengine"  # the API using Volcano engine interface
INTERFACE_TYPE_CUSTOM = "custom"  # the custom interface

# models using OpenAI interface format (including NVIDIA and native OpenAI)
OPENAI_INTERFACE_MODELS = {
    # NVIDIA models - using OpenAI interface format but using NVIDIA's service endpoint
    "meta/llama-3.1-405b-instruct": {
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None, #your own api key
        "provider": "nvidia"
    },
    
    "mistralai/mixtral-8x22b-instruct-v0.1": {
        "temperature": 0.5,
        "top_p": 1.0,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },
    
    "deepseek-ai/deepseek-r1": {
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },

    "deepseek-ai/deepseek-r1-distill-qwen-32b": {
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },

    "qwen/qwq-32b": {
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },

    
    "deepseek-ai/deepseek-r1-distill-qwen-14b": {
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },
    
    "deepseek-ai/deepseek-r1-distill-llama-8b": {
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },
    
    "deepseek-ai/deepseek-r1-distill-qwen-7b": {
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },
    
    "meta/llama-3.1-8b-instruct": {
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },  

    "qwen/qwen2.5-7b-instruct": {
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },  
    
    "google/gemma-7b": {
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },

    "mistralai/mistral-7b-instruct-v0.3": {
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": None,
        "provider": "nvidia"
    },

    "mistralai/mixtral-8x7b-instruct-v0.1": {
    "temperature": 0.5,
    "top_p": 1,
    "max_tokens": 20000,
    "stream": False,
    "base_url": "https://integrate.api.nvidia.com/v1",
    "api_key": None,
    "provider": "nvidia"
    },
    
    "gpt-4o": {
        "temperature": 0.6, 
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://api.openai.com/v1",
        "api_key": None,
        "provider": "openai"
    },
    
    "o1": {
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "base_url": "https://api.openai.com/v1",
        "api_key": None,
        "provider": "openai"
    }
    
    # can add more models using OpenAI interface here
}

# models using Volcano engine interface
VOLCENGINE_INTERFACE_MODELS = {
    "volcengine-deepseek-v3": {
        "temperature": 0.6, 
        "top_p": 0.7,
        "max_tokens": 20000,
        "stream": False,
        "model_id": "ep-20250214145915-t2msm",
        "api_key": None,
        "provider": "volcengine"
    }
    # can add more models using Volcano engine interface here
}

# custom interface model template (users can extend)
CUSTOM_INTERFACE_MODELS = {
    # example: custom model
    "my-custom-model": {
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 20000,
        "base_url": "https://api.example.com/v1",
        "api_key": None,
        "provider": "custom",
        "client_type": "openai",  
        "custom_module": None,    
        "custom_class": None,     
        "extra_args": {}          
    },

    # you can add more custom models following the above format
}

# general default parameters (if no specific model configuration)
DEFAULT_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.7,
    "max_tokens": 20000,
    "stream": False,
    "base_url": None,
    "api_key": None,
    "provider": None
}

def get_interface_type(model_name):
    """
    determine the interface type used by the model
    
    Args:
        model_name: the model name
        
    Returns:
        str: the interface type - 'openai', 'volcengine' or 'custom'
    """
    if not model_name:
        raise ValueError("Model name cannot be empty")
        
    model_name_lower = model_name.lower()
    
    # first check if the original case model name is directly in the configuration
    if model_name in VOLCENGINE_INTERFACE_MODELS:
        return INTERFACE_TYPE_VOLCENGINE
    if model_name in OPENAI_INTERFACE_MODELS:
        return INTERFACE_TYPE_OPENAI
    if model_name in CUSTOM_INTERFACE_MODELS:
        return INTERFACE_TYPE_CUSTOM
    
    
    if model_name_lower.startswith("volcengine-") or any(m.lower() == model_name_lower for m in VOLCENGINE_INTERFACE_MODELS.keys()):
        return INTERFACE_TYPE_VOLCENGINE
    
    if any(m.lower() == model_name_lower for m in OPENAI_INTERFACE_MODELS.keys()):
        return INTERFACE_TYPE_OPENAI
    
    if any(m.lower() == model_name_lower for m in CUSTOM_INTERFACE_MODELS.keys()):
        return INTERFACE_TYPE_CUSTOM
    
    if "/" in model_name_lower:
        for m in OPENAI_INTERFACE_MODELS.keys():
            if m.lower() == model_name_lower:
                return INTERFACE_TYPE_OPENAI
    
    return INTERFACE_TYPE_OPENAI

def get_model_params(model_name):

    if not model_name:
        return DEFAULT_PARAMS.copy()
        
    if model_name in OPENAI_INTERFACE_MODELS:
        return OPENAI_INTERFACE_MODELS[model_name].copy()
    
    if model_name in VOLCENGINE_INTERFACE_MODELS:
        return VOLCENGINE_INTERFACE_MODELS[model_name].copy()
    
    if model_name in CUSTOM_INTERFACE_MODELS:
        return CUSTOM_INTERFACE_MODELS[model_name].copy()
    
    model_name_lower = model_name.lower()
    interface_type = get_interface_type(model_name)
    
    if interface_type == INTERFACE_TYPE_OPENAI:
        for key, params in OPENAI_INTERFACE_MODELS.items():
            if key.lower() == model_name_lower:
                return params.copy()
    
    elif interface_type == INTERFACE_TYPE_VOLCENGINE:
        volcengine_name = model_name_lower
        if not volcengine_name.startswith("volcengine-"):
            volcengine_name = f"volcengine-{model_name_lower}"
            
        for key, params in VOLCENGINE_INTERFACE_MODELS.items():
            if key.lower() == volcengine_name:
                return params.copy()
            if key.lower().replace("volcengine-", "") == model_name_lower.replace("volcengine-", ""):
                return params.copy()
    
    elif interface_type == INTERFACE_TYPE_CUSTOM:
        for key, params in CUSTOM_INTERFACE_MODELS.items():
            if key.lower() == model_name_lower:
                return params.copy()
    
    import logging
    logging.warning(f"no configuration found for model '{model_name}', using default parameters. please add the model configuration in model_configs.py")
    
    # 返回包含接口类型的默认参数
    default_params = DEFAULT_PARAMS.copy()
    default_params["interface_type"] = interface_type
    return default_params

def get_base_url(model_name, default_url=None):

    params = get_model_params(model_name)
    return params.get("base_url") or default_url

def get_api_key(model_name, default_key=None, env_var_name=None):

    import os
    
    params = get_model_params(model_name)
    api_key = params.get("api_key")
    
    if api_key:
        return api_key
    
    provider = params.get("provider")
    
    if env_var_name and os.environ.get(env_var_name):
        return os.environ.get(env_var_name)
    
    if provider == "nvidia":
        return os.environ.get("NVIDIA_API_KEY") or default_key
    elif provider == "openai":
        return os.environ.get("OPENAI_API_KEY") or default_key
    elif provider == "volcengine":
        return os.environ.get("VOLCENGINE_API_KEY") or default_key
    
    return default_key

def get_model_id(model_name):

    if not model_name.lower().startswith("volcengine-") and get_interface_type(model_name) == INTERFACE_TYPE_VOLCENGINE:
        volcengine_name = f"volcengine-{model_name}"
    else:
        volcengine_name = model_name
    
    params = get_model_params(volcengine_name)
    
    return params.get("model_id", model_name)

def init_api_client(model_name, custom_api_key=None, custom_base_url=None):

    try:
        interface_type = get_interface_type(model_name)
        
        model_params = get_model_params(model_name)
        
        api_key = custom_api_key or get_api_key(model_name)
        if not api_key and interface_type != INTERFACE_TYPE_CUSTOM:  # 自定义接口可能不需要API密钥
            raise ValueError(f"No API key found for model {model_name}. Please provide an API key.")
        
        if interface_type == INTERFACE_TYPE_VOLCENGINE:
            from volcenginesdkarkruntime import Ark
            
            import os
            os.environ["ARK_API_KEY"] = api_key
            
            return Ark(
                api_key=api_key,
                timeout=1800  
            )
            
        elif interface_type == INTERFACE_TYPE_OPENAI:
            from openai import OpenAI
            
            base_url = custom_base_url or get_base_url(model_name)
            if not base_url:
                raise ValueError(f"No base URL found for model {model_name}. Please provide a base URL.")
            
            return OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
        elif interface_type == INTERFACE_TYPE_CUSTOM:
            client_type = model_params.get("client_type", "openai")
            
            if client_type == "openai":
                from openai import OpenAI
                base_url = custom_base_url or model_params.get("base_url")
                return OpenAI(api_key=api_key, base_url=base_url)
                
            elif client_type == "volcengine":
                from volcenginesdkarkruntime import Ark
                return Ark(api_key=api_key, timeout=1800)
                
            elif client_type == "custom_module":
                custom_module = model_params.get("custom_module")
                custom_class = model_params.get("custom_class")
                extra_args = model_params.get("extra_args", {})
                
                if not custom_module or not custom_class:
                    raise ValueError(f"Custom module and class must be specified for model {model_name}")
                
                import importlib
                module = importlib.import_module(custom_module)
                client_class = getattr(module, custom_class)
                
                client_args = {"api_key": api_key}
                if custom_base_url:
                    client_args["base_url"] = custom_base_url
                client_args.update(extra_args)
                
                return client_class(**client_args)
            
            else:
                raise ValueError(f"Unknown client type '{client_type}' for custom interface model {model_name}")
        
        else:
            raise ValueError(f"Unknown interface type for model {model_name}.")
    
    except ImportError as e:
        if "volcenginesdkarkruntime" in str(e):
            raise ImportError("Failed to import Volcengine Ark SDK. Please install with: pip install 'volcengine-python-sdk[ark]'")
        elif "openai" in str(e):
            raise ImportError("Failed to import OpenAI SDK. Please install with: pip install openai")
        else:
            raise
    
    except Exception as e:
        import traceback
        print(f"Error initializing API client for model {model_name}: {str(e)}")
        print(traceback.format_exc())
        raise

determine_api_type = get_interface_type

NVIDIA_MODELS = {k: v for k, v in OPENAI_INTERFACE_MODELS.items() if v.get("provider") == "nvidia"}
OPENAI_MODELS = {k: v for k, v in OPENAI_INTERFACE_MODELS.items() if v.get("provider") == "openai"}
VOLCENGINE_MODELS = VOLCENGINE_INTERFACE_MODELS