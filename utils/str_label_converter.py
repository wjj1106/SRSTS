from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch


class StrLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True, max_text_len=25):
        self._ignore_case = ignore_case
        self.max_text_len = max_text_len
        if self._ignore_case:
            temp = alphabet.lower()
            alphabet = []
            for c in temp:
                if c not in alphabet:
                    alphabet.append(c)
            alphabet = ''.join(alphabet)  # 得到的alpha的小写的'''

        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

        self.nClasses = len(self.dict) + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        text = [
            self.dict[char.lower() if self._ignore_case else char]
            for char in text
        ]
        length = min(len(text), self.max_text_len)
        text = text + [0] * (self.max_text_len - length)

        return text, length

    def decode(self, t, length, scores=None, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]

            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                temp_score = []
                import numpy as np
                all_scores = np.zeros((self.max_text_len, len(self.alphabet)))
                
                t_length = 0
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[int(t[i]) - 1])
                        temp_score.append(scores[i][t[i]].item())
                        all_scores[t_length] = scores[i].cpu().detach().numpy()
                        t_length += 1
                res = ''.join(char_list)

                return res, sum(temp_score) / t_length if len(res) > 0 else 0, all_scores
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
