## SAMPLE EFFICIENT ADAPTIVE TEXT-TO-SPEECH

Plan to implement the ICLR2019 paper: SEA-TTS

# How to
* CBHG with speaker embedding, as local conditional of wavenet.
* Transformer of encoder module refer to deepvoice3, as every timestep of wavenet query.
* speaker embedding feed into every timestep of wavenet, as global contitional of wavenet.

