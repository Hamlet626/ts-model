import numpy as np
import torch
import nemo.collections.asr as nemo_asr
from triton_python_backend_utils import get_input_tensor_by_name, get_output_tensor_by_name #, InferenceResponse, TritonError
import triton_python_backend_utils as pb_utils
import traceback

class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """`auto_complete_config` is called only once when loading the model
        assuming the server was not started with
        `--disable-auto-complete-config`. Implementing this function is
        optional. No implementation of `auto_complete_config` will do nothing.
        This function can be used to set `max_batch_size`, `input` and `output`
        properties of the model using `set_max_batch_size`, `add_input`, and
        `add_output`. These properties will allow Triton to load the model with
        minimal model configuration in absence of a configuration file. This
        function returns the `pb_utils.ModelConfig` object with these
        properties. You can use the `as_dict` function to gain read-only access
        to the `pb_utils.ModelConfig` object. The `pb_utils.ModelConfig` object
        being returned from here will be used as the final configuration for
        the model.

        Note: The Python interpreter used to invoke this function will be
        destroyed upon returning from this function and as a result none of the
        objects created here will be available in the `initialize`, `execute`,
        or `finalize` functions.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration. You can build
          upon the configuration given by this object when setting the
          properties for this model.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        inputs = [{
            'name': 'INPUT',
            'data_type': 'TYPE_FP32',
            'dims': [-1]
        }]
        outputs = [{
            'name': 'OUTPUT',
            'data_type': 'TYPE_STRING',
            'dims': [1]
        }]

        # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_input` will check for conflicts and
            # raise errors if an input with the same name already exists in
            # the configuration but has different data_type or dims property.
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_output` will check for conflicts and
            # raise errors if an output with the same name already exists in
            # the configuration but has different data_type or dims property.
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        # auto_complete_model_config.set_max_batch_size(0)

        # To enable a dynamic batcher with default settings, you can use
        # auto_complete_model_config set_dynamic_batching() function. It is
        # commented in this example because the max_batch_size is zero.
        #
        auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config
    
    def initialize(self, args):
        # Load the model
        self.model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_ctc_large")

    def execute(self, requests):
        logger = pb_utils.Logger
        logger.log_warn('degug execute running!>>!')
        responses = []
        logger.log_info(str(requests))
        try:
            # Get the input tensor
            input_tensor = [get_input_tensor_by_name(request, "INPUT").as_numpy() for request in requests]
            
            logger.log_info(str(input_tensor))
            logger.log_info(f"Data: {input_tensor[0]}")
        
            # responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUTPUT", np.array(['transcription'], dtype=np.str_))]))
            # return responses
        
            audio_data = np.stack(input_tensor, axis=0)

            logger.log_info(str(audio_data.shape))
            # Preprocess the audio (resample, convert to mono, etc.)
            audio_tensor = torch.as_tensor(audio_data, dtype=torch.float32).to(self.model.device)
            input_signal_length = torch.tensor([d.shape[0] for d in audio_data], dtype=torch.int64).to(self.model.device)

            logger.log_info(str(audio_tensor.shape))
            logger.log_info(str(input_signal_length))
        
            # Perform inference
            with torch.no_grad():
                _, _, predictions = self.model(input_signal=audio_tensor, input_signal_length=input_signal_length)

            # Post-process the output
            blank_id = len(self.model.decoder.vocabulary)

            # responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUTPUT", np.array(['transcription'], dtype=np.object_))]))
            # return responses
        
            for i in range(predictions.cpu().numpy().shape[0]):
                predictions = [tok for tok in predictions.cpu().numpy()[i] if tok != blank_id]
                transcription = self.model.tokenizer.ids_to_text(list(map(int, predictions)))

                # Prepare the output
                out_tensor = pb_utils.Tensor("OUTPUT", np.array([transcription], dtype=np.object_))

                # output_tensor = get_output_tensor_by_name(requests[i], "OUTPUT")
                # output_tensor.set_data_from_numpy(np.array([transcription]))

                # Add the response
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        except Exception as e:
            error_message = ''.join(traceback.format_exception(None, e, e.__traceback__))
            logger.log_warn(str(e))
            logger.log_warn(error_message)
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUTPUT", np.array([error_message], dtype=np.str_))]))
            # raise pb_utils.TritonModelException(e)
        return responses
    
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        self.model=None
        print("Cleaning up...")