# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions for CoNLL evaluation. We can add other datasets conversion
function for CoNLL here in the future.
"""


def write_tokens_to_file(pred_pack, pred_request, refer_pack, refer_request,
                         output_filename):
    r"""Write the predictions to a file on which CoNLL Eval script is run.

    Args:
        pred_pack (DataPack): Predicted data pack
        pred_request (Dict): A data request to fetch relevant data from the
            predict pack
        refer_pack (DataPack): Gold data pack i.e. pack containing gold labels
        refer_request (Dict): A data request to fetch relevant data from the
            gold pack
        output_filename (str): Filename to write outputs to
    """
    opened_file = open(output_filename, "w+")
    for pred_sentence, tgt_sentence in zip(
            pred_pack.get_data(**pred_request),
            refer_pack.get_data(**refer_request)):

        pred_tokens, tgt_tokens = (pred_sentence["Token"],
                                   tgt_sentence["Token"])

        for i in range(len(pred_tokens["text"])):
            w = tgt_tokens["text"][i]
            p = tgt_tokens["pos"][i]
            ch = tgt_tokens["chunk"][i]
            tgt = tgt_tokens["pos"][i]
            pred = pred_tokens["pos"][i]

            opened_file.write(
                "%d %s %s %s %s %s\n" % (i + 1, w, p, ch, tgt, pred))
        opened_file.write("\n")

    opened_file.close()
