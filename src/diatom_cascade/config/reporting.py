"""
Utility function to print training configuration in a concise table format
"""

def print_training_config(model_type, config_class, training_config):
    """
    Print training configuration in a concise table format, organized by three categories.
    
    Args:
        model_type: Model type string (e.g., "F-C", "H-COF")
        config_class: Local Config class with model-specific settings
        training_config: TrainAndValConfig class with all hyperparameters
    """
    print(f"\n{'='*70}")
    print(f"Training Configuration: {model_type}")
    print(f"{'='*70}")
    
    # Category 1: Architecture-Independent Parameters
    print("\n[Category 1] Architecture-Independent (Same for all models):")
    print(f"  {'Parameter':<25} {'Value':<20}")
    print(f"  {'-'*25} {'-'*20}")
    print(f"  {'IMAGE_SIZE':<25} {training_config.IMAGE_SIZE:<20}")
    print(f"  {'BATCH_SIZE':<25} {training_config.BATCH_SIZE:<20}")
    print(f"  {'BASE_MODEL':<25} {training_config.BASE_MODEL:<20}")
    
    # Show pretrained model info if available (for hierarchical models)
    pretrained_info = None
    if hasattr(config_class, 'CLASS_MODEL_PATH'):
        pretrained_path = config_class.CLASS_MODEL_PATH
        if hasattr(pretrained_path, 'name'):
            pretrained_info = f"from {pretrained_path.name}"
        else:
            pretrained_info = f"from {str(pretrained_path)}"
    elif hasattr(config_class, 'PRETRAINED_MODEL_PATH'):
        pretrained_path = config_class.PRETRAINED_MODEL_PATH
        if hasattr(pretrained_path, 'name'):
            pretrained_info = f"from {pretrained_path.name}"
        else:
            pretrained_info = f"from {str(pretrained_path)}"
    
    if pretrained_info:
        print(f"  {'BACKBONE_WEIGHTS':<25} {pretrained_info:<20}")
    
    print(f"  {'OPTIMIZER':<25} {training_config.OPTIMIZER:<20}")
    print(f"  {'WEIGHT_DECAY':<25} {training_config.WEIGHT_DECAY:<20}")
    print(f"  {'MAX_EPOCHS':<25} {training_config.MAX_EPOCHS:<20} (was 50)")
    print(f"  {'RANDOM_SEED':<25} {training_config.RANDOM_SEED:<20}")
    print(f"  {'NUM_WORKERS':<25} {training_config.NUM_WORKERS:<20}")
    
    # Category 2: Architecture-Dependent Parameters
    print("\n[Category 2] Architecture-Dependent (Unified start):")
    print(f"  {'Parameter':<25} {'Value':<20}")
    print(f"  {'-'*25} {'-'*20}")
    print(f"  {'INITIAL_LR':<25} {training_config.INITIAL_LR:<20}")
    print(f"  {'LR_SCHEDULER':<25} {training_config.LR_SCHEDULER_TYPE:<20} (was CosineAnnealingLR)")
    lr_config = training_config.LR_SCHEDULER_CONFIG
    print(f"  {'  - mode':<25} {lr_config.get('mode', 'N/A'):<20}")
    print(f"  {'  - factor':<25} {lr_config.get('factor', 'N/A'):<20}")
    print(f"  {'  - patience':<25} {lr_config.get('patience', 'N/A'):<20}")
    print(f"  {'  - min_lr':<25} {lr_config.get('min_lr', 'N/A'):<20}")
    print(f"  {'EARLY_STOP_PATIENCE':<25} {training_config.EARLY_STOPPING_PATIENCE:<20} (was 10)")
    print(f"  {'EARLY_STOP_MIN_DELTA':<25} {training_config.EARLY_STOPPING_MIN_DELTA:<20}")
    
    # Category 3: Task-Specific Parameters
    print("\n[Category 3] Task-Specific (Model-dependent):")
    print(f"  {'Parameter':<25} {'Value':<20}")
    print(f"  {'-'*25} {'-'*20}")
    print(f"  {'TRAIN_RATIO':<25} {training_config.TRAIN_RATIO:<20}")
    print(f"  {'VAL_RATIO':<25} {training_config.VAL_RATIO:<20}")
    print(f"  {'TEST_RATIO':<25} {training_config.TEST_RATIO:<20}")
    print(f"  {'STRATIFY_BY':<25} {training_config.STRATIFY_BY.get(model_type, 'N/A'):<20}")
    print(f"  {'FOCAL_ALPHA':<25} {training_config.FOCAL_ALPHA:<20}")
    print(f"  {'FOCAL_GAMMA':<25} {training_config.FOCAL_GAMMA:<20}")
    
    # Model-specific parameters
    if hasattr(config_class, 'NUM_CLASSES'):
        print(f"  {'NUM_CLASSES':<25} {config_class.NUM_CLASSES:<20}")
    if hasattr(config_class, 'NUM_ORDERS'):
        print(f"  {'NUM_ORDERS':<25} {config_class.NUM_ORDERS:<20}")
    if hasattr(config_class, 'NUM_FAMILIES'):
        print(f"  {'NUM_FAMILIES':<25} {config_class.NUM_FAMILIES:<20}")
    if hasattr(config_class, 'NUM_GENERA'):
        print(f"  {'NUM_GENERA':<25} {config_class.NUM_GENERA:<20}")
    if hasattr(config_class, 'NUM_SPECIES'):
        print(f"  {'NUM_SPECIES':<25} {config_class.NUM_SPECIES:<20}")
    
    # Loss weights for hierarchical models
    if model_type in training_config.LOSS_WEIGHTS:
        print(f"  {'LOSS_WEIGHTS':<25} {str(training_config.LOSS_WEIGHTS[model_type]):<20}")
    
    # Dropout
    if model_type.startswith('F-'):
        dropout = training_config.DROPOUT_FLAT
    elif model_type == 'H-CO':
        dropout = training_config.DROPOUT_HEAD_2LAYER
    elif model_type == 'H-COF':
        dropout = training_config.DROPOUT_HEAD_3LAYER
    elif model_type == 'H-COFG':
        dropout = training_config.DROPOUT_HEAD_4LAYER
    elif model_type == 'H-COFGS':
        dropout = training_config.DROPOUT_HEAD_4LAYER
    else:
        dropout = 'N/A'
    print(f"  {'DROPOUT':<25} {str(dropout):<20}")
    
    # Prediction method (from Config class)
    if hasattr(config_class, 'PREDICTION_METHOD'):
        prediction_method = config_class.PREDICTION_METHOD
    else:
        # Fallback: infer from model type if not explicitly set
        if model_type.startswith('F-'):
            prediction_method = 'torch.argmax'
        elif model_type.startswith('H-'):
            prediction_method = 'greedy_hierarchical_predict'
        else:
            prediction_method = 'N/A'
    print(f"  {'PREDICTION_METHOD':<25} {prediction_method:<20}")
    
    print(f"{'='*70}\n")

