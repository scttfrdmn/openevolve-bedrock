# Copyright 2026 Scott Friedman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
openevolve-bedrock: Amazon Bedrock as the LLM backend for OpenEvolve.

Importing this package registers the ``bedrock`` provider with OpenEvolve's
LLM ensemble. In a config.yaml::

    llm:
      provider: bedrock
      api_base: aws://us-west-2          # Converse API, SigV4 (or Bedrock API key)
      models:
        - name: us.amazon.nova-2-lite-v1:0
          weight: 0.8
        - name: us.anthropic.claude-sonnet-5-5
          weight: 0.2

or, for the OpenAI-compatible Mantle endpoint::

    llm:
      provider: bedrock
      api_base: https://bedrock-mantle.us-east-1.api.aws/v1
      api_key: "${AWS_BEARER_TOKEN_BEDROCK}"
"""

from openevolve_bedrock.provider import BedrockLLM, register

__version__ = "0.1.0"
__all__ = ["BedrockLLM", "register", "__version__"]

register()
