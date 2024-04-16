import torch
from torch.nn import ReLU
import torch.nn.functional as F

from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.children())[0]
        first_layer.register_full_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        #model_output = self.model(input_image)
        #print("Model output:", model_output)
        # Zero gradients
        #self.model.zero_grad()
        #assert target_class < model_output.size()[-1], "Target class out of range."
        # Target for backprop
        #one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        #one_hot_output[0][target_class] = 1
        #one_hot_output = one_hot_output.to(model_output.device)

        # Backward pass
        #model_output.backward(gradient=one_hot_output)

        input_image.requires_grad_(True)
            # Forward
        model_output = self.model(input_image) 
        print('Forward pass executed.')
        print('Model output shape:', model_output.shape)
        # We need to ensure the output is a scalar for binary classification
        # In binary classification, target_class is usually 0 or 1
        self.model.zero_grad()


        model_output = self.model(input_image)
        print("Logits: ", model_output)
        print("Logits shape: ", model_output.shape)
        print("Logits data type: ", model_output.dtype)
        print("Logits range: min=", model_output.min().item(), ", max=", model_output.max().item())
        


        # Backward pass
        target = torch.tensor([[float(target_class)]], requires_grad=False)

        print("Target labels: ", target)
        print("Target data type: ", target.dtype)
        print('Target shape:', target.shape)
        # We use binary_cross_entropy_with_logits to create a scalar loss
        # as it expects the raw scores without sigmoid applied.
        loss = F.binary_cross_entropy_with_logits(model_output, target)
        print('Loss computed:', loss)
        loss.backward()

        print('Backward pass executed.')
        print('Gradients captured:', self.gradients is not None)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(name, "Gradient max:", param.grad.max().item(), "Gradient min:", param.grad.min().item())
        else:
            print(name, "No gradient")
        if prep_img.grad is not None:
            print('Input image gradients captured:', prep_img.grad.size())
            # Now we know we have gradients, ensure they are not zero
            if not torch.any(prep_img.grad):
                print('Gradients are zero. There might be an issue with the loss or backward pass.')
        else:
            print('No gradients in input image.')
        print('Gradients shape:', self.gradients.shape)
        gradients_as_arr = self.gradients.data.numpy()[0]
        print('Gradients as array shape:', gradients_as_arr.shape)
        return gradients_as_arr



if __name__ == '__main__':
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example)

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    print(pos_sal)
    print(neg_sal)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')
