# Testing Guide

<cite>
**Referenced Files in This Document**
- [test_forward_minimal.py](file://tests/test_forward_minimal.py)
- [test_forward_simple.py](file://tests/test_forward_simple.py)
- [test_model_forward_inputs.py](file://tests/test_model_forward_inputs.py)
- [test_tokenizer_smoke.py](file://tests/test_tokenizer_smoke.py)
- [test_refactoring_syntax.py](file://tests/test_refactoring_syntax.py)
- [config.yaml](file://configs/config.yaml)
- [requirements.txt](file://requirements.txt)
- [README.md](file://README.md)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Testing Framework Overview](#testing-framework-overview)
3. [Core Testing Components](#core-testing-components)
4. [Forward Pass Testing](#forward-pass-testing)
5. [Tokenizer Testing](#tokenizer-testing)
6. [Refactoring Validation](#refactoring-validation)
7. [Integration Testing](#integration-testing)
8. [Testing Best Practices](#testing-best-practices)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Conclusion](#conclusion)

## Introduction

This Testing Guide provides comprehensive documentation for the GraphGPT testing framework, covering forward pass validation, tokenizer functionality, refactoring verification, and integration testing approaches. The guide focuses on the four primary test files that validate different aspects of the GraphGPT system architecture.

## Testing Framework Overview

The GraphGPT testing framework consists of four main categories of tests that validate different components of the system:

```mermaid
graph TB
subgraph "Testing Categories"
A[Forward Pass Tests]
B[Tokenizer Tests]
C[Refactoring Tests]
D[Integration Tests]
end
subgraph "Test Files"
A1[test_forward_minimal.py]
A2[test_forward_simple.py]
A3[test_model_forward_inputs.py]
B1[test_tokenizer_smoke.py]
C1[test_refactoring_syntax.py]
end
A --> A1
A --> A2
A --> A3
B --> B1
C --> C1
D --> A1
D --> A2
D --> A3
D --> B1
D --> C1
```

**Section sources**
- [test_forward_minimal.py:1-568](file://tests/test_forward_minimal.py#L1-L568)
- [test_forward_simple.py:1-351](file://tests/test_forward_simple.py#L1-L351)
- [test_model_forward_inputs.py:1-457](file://tests/test_model_forward_inputs.py#L1-L457)
- [test_tokenizer_smoke.py:1-232](file://tests/test_tokenizer_smoke.py#L1-L232)
- [test_refactoring_syntax.py:1-265](file://tests/test_refactoring_syntax.py#L1-L265)

## Core Testing Components

### Forward Pass Testing Suite

The forward pass testing suite validates model input structures and tensor shapes across different training modes:

```mermaid
classDiagram
class ForwardPassTests {
+minimal_forward_test()
+simple_forward_test()
+advanced_inspection_test()
+print_tensor_info()
+validate_shapes()
+check_dtypes()
}
class MinimalTest {
+print_section()
+describe_tensor()
+pretrain_mode_inputs()
+finetune_mode_inputs()
}
class SimpleTest {
+test_pretrain_forward()
+test_finetune_forward()
+test_collator_batch_structure()
+create_dummy_data()
}
class AdvancedTest {
+ForwardInputInspector
+attach_to_model()
+inspect_forward_inputs()
+setup_config()
}
ForwardPassTests --> MinimalTest
ForwardPassTests --> SimpleTest
ForwardPassTests --> AdvancedTest
```

**Diagram sources**
- [test_forward_minimal.py:13-140](file://tests/test_forward_minimal.py#L13-L140)
- [test_forward_simple.py:40-254](file://tests/test_forward_simple.py#L40-L254)
- [test_model_forward_inputs.py:39-122](file://tests/test_model_forward_inputs.py#L39-L122)

### Tokenizer Testing Framework

The tokenizer testing framework validates import functionality, utility functions, and public API surfaces:

```mermaid
classDiagram
class TokenizerTests {
+ImportTests
+UtilityTests
+RegistrationTests
+ModuleTests
+test_import_resolutions()
+test_function_callable()
+test_constant_values()
+test_task_registration()
}
class ImportTests {
+test_import_tokenizer_classes()
+test_import_collator()
+test_import_utilities()
+test_import_vocab_builder()
}
class UtilityTests {
+test_tokenization_output()
+test_mask_functions()
+test_input_preparation()
+test_module_functions()
}
class RegistrationTests {
+test_known_task_types()
+test_unknown_task_type()
+test_task_type_resolution()
}
TokenizerTests --> ImportTests
TokenizerTests --> UtilityTests
TokenizerTests --> RegistrationTests
```

**Diagram sources**
- [test_tokenizer_smoke.py:22-232](file://tests/test_tokenizer_smoke.py#L22-L232)

### Refactoring Validation Tests

The refactoring validation tests ensure code structure integrity and backward compatibility:

```mermaid
classDiagram
class RefactoringTests {
+SyntaxValidation
+ClassStructure
+CompositionPattern
+BackwardCompatibility
+test_syntax_validity()
+test_class_hierarchy()
+test_composition_patterns()
+test_backward_compatibility()
}
class SyntaxValidation {
+test_base_tokenizer_syntax()
+test_core_tokenizer_syntax()
+test_strategy_syntax()
+test_init_syntax()
}
class ClassStructure {
+test_base_tokenizer_abstract()
+test_inheritance_chains()
+test_abstract_methods()
+test_strategy_inheritance()
}
class CompositionPattern {
+test_attribute_presence()
+test_strategy_attributes()
+test_composition_usage()
}
class BackwardCompatibility {
+test_legacy_exports()
+test_new_exports()
+test_api_compatibility()
}
RefactoringTests --> SyntaxValidation
RefactoringTests --> ClassStructure
RefactoringTests --> CompositionPattern
RefactoringTests --> BackwardCompatibility
```

**Diagram sources**
- [test_refactoring_syntax.py:14-265](file://tests/test_refactoring_syntax.py#L14-L265)

**Section sources**
- [test_forward_minimal.py:1-568](file://tests/test_forward_minimal.py#L1-L568)
- [test_forward_simple.py:1-351](file://tests/test_forward_simple.py#L1-L351)
- [test_model_forward_inputs.py:1-457](file://tests/test_model_forward_inputs.py#L1-L457)
- [test_tokenizer_smoke.py:1-232](file://tests/test_tokenizer_smoke.py#L1-L232)
- [test_refactoring_syntax.py:1-265](file://tests/test_refactoring_syntax.py#L1-L265)

## Forward Pass Testing

### Minimal Forward Pass Testing

The minimal forward pass test provides comprehensive documentation of model input structures without requiring full environment setup:

```mermaid
sequenceDiagram
participant User as "User"
participant Script as "test_forward_minimal.py"
participant Console as "Console Output"
User->>Script : Run minimal forward test
Script->>Console : Print model architecture info
Script->>Console : Describe pre-training inputs
Script->>Console : Describe fine-tuning inputs
Script->>Console : Explain data flow process
Script->>Console : Show configuration impact
Script->>Console : Provide debugging tips
Script->>Console : Display summary
Console-->>User : Complete forward pass documentation
```

**Diagram sources**
- [test_forward_minimal.py:32-561](file://tests/test_forward_minimal.py#L32-L561)

### Simple Forward Pass Testing

The simple forward pass test executes actual model forward passes with dummy data:

```mermaid
flowchart TD
Start([Start Simple Test]) --> CreateConfig["Create Minimal Config"]
CreateConfig --> CreateModel["Create GraphGPT Model"]
CreateModel --> CreateDummyData["Generate Dummy Input Data"]
CreateDummyData --> PretrainForward["Execute Pre-training Forward Pass"]
PretrainForward --> FinetuneForward["Execute Fine-tuning Forward Pass"]
FinetuneForward --> CollatorTest["Test Collator Output"]
CollatorTest --> ValidateResults["Validate Tensor Shapes & Values"]
ValidateResults --> End([End Test])
```

**Diagram sources**
- [test_forward_simple.py:40-346](file://tests/test_forward_simple.py#L40-L346)

### Advanced Forward Input Inspection

The advanced testing framework uses hook-based inspection for detailed input analysis:

```mermaid
classDiagram
class ForwardInputInspector {
+captured_inputs : List
+original_forward : Callable
+model : nn.Module
+attach(model)
+detach()
+_print_inputs(captured)
}
class TrainingPipeline {
+model : nn.Module
+mode : TrainingMode
+setup_config()
+prepare_data()
+create_model()
+setup_training()
}
class TrainingUtils {
+batch_training()
+ft_batch_training()
}
ForwardInputInspector --> TrainingPipeline : "hooks model"
TrainingPipeline --> TrainingUtils : "calls during training"
```

**Diagram sources**
- [test_model_forward_inputs.py:39-122](file://tests/test_model_forward_inputs.py#L39-L122)
- [test_model_forward_inputs.py:180-344](file://tests/test_model_forward_inputs.py#L180-L344)

**Section sources**
- [test_forward_minimal.py:1-568](file://tests/test_forward_minimal.py#L1-L568)
- [test_forward_simple.py:1-351](file://tests/test_forward_simple.py#L1-L351)
- [test_model_forward_inputs.py:1-457](file://tests/test_model_forward_inputs.py#L1-L457)

## Tokenizer Testing

### Import Resolution Testing

The tokenizer smoke tests validate that all public import paths resolve correctly:

```mermaid
graph LR
subgraph "Import Paths"
A[src.data.tokenizer]
B[src.data.collator]
C[src.utils.tokenizer_utils]
D[src.data.vocab_builder]
end
subgraph "Test Classes"
E[ImportResolutionTests]
F[PublicAPITests]
G[UtilityFunctionTests]
end
A --> E
B --> E
C --> F
D --> F
E --> G
```

**Diagram sources**
- [test_tokenizer_smoke.py:22-77](file://tests/test_tokenizer_smoke.py#L22-L77)

### Tokenization Output Validation

The tokenization output tests verify TokenizationOutput functionality and field mutations:

```mermaid
classDiagram
class TokenizationOutput {
+ls_tokens : List[str]
+ls_labels : List[str]
+__init__(ls_tokens=None, ls_labels=None)
+field_mutation_tests()
}
class MaskFunctionTests {
+test_basic_mask_first()
+test_mask_all()
+test_single_element()
+get_mask_of_raw_seq()
}
class InputPreparationTests {
+test_basic_with_labels()
+test_with_label_padding()
+test_autoregressive_mode()
+get_input_dict_from_seq_tokens_id()
}
TokenizationOutput --> MaskFunctionTests
TokenizationOutput --> InputPreparationTests
```

**Diagram sources**
- [test_tokenizer_smoke.py:84-177](file://tests/test_tokenizer_smoke.py#L84-L177)

### Task Type Registration Testing

The registration tests validate task-type functionality across different task categories:

```mermaid
graph TB
subgraph "Known Task Types"
A[pretrain]
B[pretrain-mlm]
C[pretrain-coord]
D[graph]
E[edge]
F[node]
G[nodev2]
end
subgraph "Registration Tests"
H[RegistrationValidation]
I[CallableVerification]
J[UnknownTypeHandling]
end
A --> H
B --> H
C --> H
D --> H
E --> H
F --> H
G --> H
H --> I
H --> J
```

**Diagram sources**
- [test_tokenizer_smoke.py:179-206](file://tests/test_tokenizer_smoke.py#L179-L206)

**Section sources**
- [test_tokenizer_smoke.py:1-232](file://tests/test_tokenizer_smoke.py#L1-L232)

## Refactoring Validation

### Syntax Validation Testing

The refactoring syntax tests ensure all refactored files maintain valid Python syntax:

```mermaid
flowchart TD
Start([Start Syntax Validation]) --> ParseFiles["Parse Refactored Files"]
ParseFiles --> ValidateBase["Validate Base Tokenizer"]
ValidateBase --> ValidateCore["Validate Core Tokenizer"]
ValidateCore --> ValidateStrategies["Validate Strategy Files"]
ValidateStrategies --> ValidateInit["Validate __init__.py Files"]
ValidateInit --> CheckAbstract["Check Abstract Base Classes"]
CheckAbstract --> VerifyInheritance["Verify Inheritance Chains"]
VerifyInheritance --> ValidateExports["Validate Exports"]
ValidateExports --> End([End Validation])
```

**Diagram sources**
- [test_refactoring_syntax.py:14-61](file://tests/test_refactoring_syntax.py#L14-L61)

### Class Structure Validation

The class structure tests verify inheritance patterns and abstract method implementations:

```mermaid
classDiagram
class BaseTokenizer {
<<abstract>>
+padding_strategy : Strategy
+task_preparer : Strategy
+sequence_packer : Strategy
+abstract_methods()
}
class GSTTokenizer {
+inherits : BaseTokenizer
+tokenizer_methods()
}
class StackedGSTTokenizer {
+inherits : BaseTokenizer
+stacked_tokenizer_methods()
}
class PaddingStrategy {
<<abstract>>
+flat_padding_methods()
+stacked_padding_methods()
}
class FlatPaddingStrategy {
+inherits : PaddingStrategy
}
class StackedPaddingStrategy {
+inherits : PaddingStrategy
}
BaseTokenizer <|-- GSTTokenizer
BaseTokenizer <|-- StackedGSTTokenizer
BaseTokenizer --> PaddingStrategy
PaddingStrategy <|-- FlatPaddingStrategy
PaddingStrategy <|-- StackedPaddingStrategy
```

**Diagram sources**
- [test_refactoring_syntax.py:63-165](file://tests/test_refactoring_syntax.py#L63-L165)

**Section sources**
- [test_refactoring_syntax.py:1-265](file://tests/test_refactoring_syntax.py#L1-L265)

## Integration Testing

### Configuration-Based Testing

The integration testing framework validates configuration impacts on model inputs:

```mermaid
graph LR
subgraph "Configuration Options"
A[Token Packing]
B[Flex Attention]
C[Node/Edge Embeddings]
D[DeepSpeed vs DDP]
end
subgraph "Input Impact"
E[split_lens]
F[attn_modes]
G[inputs_raw_embeds]
H[model_dtype]
end
subgraph "Testing Methods"
I[Print in Training Loop]
J[Hook on Forward Method]
K[Modify Collator]
end
A --> E
B --> F
C --> G
D --> H
E --> I
F --> J
G --> K
H --> I
```

**Diagram sources**
- [test_forward_minimal.py:334-392](file://tests/test_forward_minimal.py#L334-L392)
- [test_forward_minimal.py:399-462](file://tests/test_forward_minimal.py#L399-L462)

### Pipeline Integration Testing

The pipeline integration tests validate end-to-end training workflows:

```mermaid
sequenceDiagram
participant Config as "Configuration"
participant Pipeline as "TrainingPipeline"
participant Data as "Dataset"
participant Model as "Model"
participant Utils as "Training Utils"
Config->>Pipeline : Initialize with mode
Pipeline->>Pipeline : Extract config
Pipeline->>Data : Prepare data loaders
Pipeline->>Model : Create model instance
Pipeline->>Utils : Setup optimizer
Pipeline->>Utils : Execute training step
Utils->>Model : Forward pass with batch
Model-->>Utils : Return outputs
Utils-->>Pipeline : Training statistics
Pipeline-->>Config : Update progress
```

**Diagram sources**
- [test_model_forward_inputs.py:180-344](file://tests/test_model_forward_inputs.py#L180-L344)

**Section sources**
- [test_forward_minimal.py:1-568](file://tests/test_forward_minimal.py#L1-L568)
- [test_model_forward_inputs.py:1-457](file://tests/test_model_forward_inputs.py#L1-L457)

## Testing Best Practices

### Debugging and Troubleshooting

The testing framework provides comprehensive debugging utilities and common error patterns:

```mermaid
flowchart TD
Problem[Problem Encountered] --> CheckShapes["Check Tensor Shapes"]
CheckShapes --> VerifyDtypes["Verify Data Types"]
VerifyDtypes --> CheckDevices["Check Device Placement"]
CheckDevices --> ValidateRanges["Validate Value Ranges"]
ValidateRanges --> MonitorGradients["Monitor Gradient Norms"]
MonitorGradients --> CheckLoss["Check Loss Behavior"]
CheckShapes --> FixShapes["Fix Shape Mismatches"]
VerifyDtypes --> FixDtypes["Fix Data Type Issues"]
CheckDevices --> FixDevices["Move to Correct Device"]
ValidateRanges --> FixRanges["Clip Out-of-Range Values"]
MonitorGradients --> FixGradients["Adjust Learning Rate"]
CheckLoss --> FixLoss["Investigate Training Issues"]
FixShapes --> ReRun[Re-run Test]
FixDtypes --> ReRun
FixDevices --> ReRun
FixRanges --> ReRun
FixGradients --> ReRun
FixLoss --> ReRun
```

### Common Error Patterns

The testing framework documents common errors and their solutions:

| Error Type | Description | Solution |
|------------|-------------|----------|
| Device Mismatch | Expected device cuda but got cpu | Move batch to model device |
| Shape Mismatch | Dimension mismatch in tensor operations | Check stacked_feat, batch_size, seq_len |
| Memory Issues | CUDA out of memory errors | Reduce batch_size or max_position_embeddings |
| Label Range | Labels outside vocabulary range | Ensure labels < vocab_size or = -100 for masking |

**Section sources**
- [test_forward_minimal.py:498-526](file://tests/test_forward_minimal.py#L498-L526)

## Troubleshooting Guide

### Environment Setup Issues

Common installation and environment problems:

```mermaid
flowchart TD
Start([Environment Issue]) --> CheckPython["Check Python Version"]
CheckPython --> CheckDependencies["Check Dependencies"]
CheckDependencies --> CheckCUDA["Check CUDA Compatibility"]
CheckCUDA --> CheckDeepspeed["Check DeepSpeed Version"]
CheckPython --> InstallPython["Install Compatible Python"]
CheckDependencies --> InstallDeps["Install Required Dependencies"]
CheckCUDA --> InstallCUDA["Install Compatible CUDA"]
CheckDeepspeed --> InstallDeepspeed["Install Specific DeepSpeed Version"]
InstallPython --> TestInstall[Test Installation]
InstallDeps --> TestInstall
InstallCUDA --> TestInstall
InstallDeepspeed --> TestInstall
TestInstall --> Success([Installation Successful])
```

### Test Execution Problems

Problems encountered during test execution:

```mermaid
graph TB
subgraph "Test Execution Issues"
A[Import Errors]
B[Runtime Errors]
C[Memory Issues]
D[Configuration Errors]
end
subgraph "Solutions"
E[Check Python Path]
F[Validate Dependencies]
G[Reduce Batch Size]
H[Review Config Files]
end
A --> E
B --> F
C --> G
D --> H
```

**Section sources**
- [requirements.txt:1-28](file://requirements.txt#L1-L28)
- [README.md:203-223](file://README.md#L203-L223)

## Conclusion

The GraphGPT testing framework provides comprehensive validation across multiple testing categories, ensuring code quality, functionality, and maintainability. The framework's modular design allows for targeted testing of specific components while maintaining integration validation through the pipeline testing approach.

Key testing capabilities include:
- Forward pass validation with detailed input inspection
- Tokenizer functionality verification
- Refactoring validation for code structure integrity
- Integration testing through training pipeline validation
- Comprehensive debugging utilities and error handling

The testing framework supports both lightweight smoke tests and comprehensive integration tests, making it suitable for development, CI/CD pipelines, and production validation scenarios.
