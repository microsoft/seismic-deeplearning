"""
TrainPipeline class for setting up a training pipeline in AzureML.
Inherits from DeepSeismicAzMLPipeline
"""
from deepseismic_interpretation.azureml_pipelines.base_pipeline import DeepSeismicAzMLPipeline


class TrainPipeline(DeepSeismicAzMLPipeline):

    def construct_pipeline(self):
        """
        implemented function from ABC. Sets up the pre-requisites for a pipeline.
        """
        self._setup_compute()
        self._setup_datastore(blob_dataset_name=self.config['step1']['input_dataset_name'])

        self._setup_steps()

    def _setup_steps(self):
        """
        iterates over all the steps in the config file and sets each one up along
        with its accompanying objects.
        """
        for _, step in self.config.items():
            try:
                input_data = self._setup_datareference(name=step['input_datareference_name'],
                                                       path=step['input_datareference_path'])
            except KeyError:
                # grab the last step's output as input for this step
                if self.last_output_data is None:
                    raise KeyError("input_datareference_name and input_datareference_path can only be"
                                   "omitted if there is a previous step in the pipeline")
                else:
                    input_data = self.last_output_data

            try:
                self.last_output_data = self._setup_pipelinedata(name=step['output'],
                                                                 output_path=step.get('output_path', None))
            except KeyError:
                self.last_output_data = None

            script_params = step['arguments'] + ['--input', input_data]
            
            if self.last_output_data is not None:
                script_params = script_params + ['--output', self.last_output_data]

            self._create_pipeline_step(step=step,
                                      arguments=script_params,
                                      input_data=input_data,
                                      output=self.last_output_data)
