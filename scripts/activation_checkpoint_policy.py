"""Trade activation memory for fewer decoder recomputations in this process."""


def retain_layer_activations(model, count):
    if count == 0:
        return []
    layers = [(name, module) for name, module in model.named_modules()
              if type(module).__name__ == 'Qwen3_5DecoderLayer']
    if not 0 <= count < len(layers):
        raise ValueError('Activation retention count must be below decoder layer count')
    selected = []
    for index in range(count):
        name, layer = layers[index*len(layers)//count]
        if not layer.gradient_checkpointing:
            raise ValueError('Expected enabled gradient checkpointing before retention policy')
        layer.gradient_checkpointing = False
        selected.append(name)
    return selected
