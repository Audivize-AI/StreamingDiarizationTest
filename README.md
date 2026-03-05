This branch is a Swift test for NVIDIA's Streaming Sortformer diarizer as a replacement for FluidAudio's old diarizer. 

This version introduces NVIDIA's TitaNet speaker verification model for extremely lightweight speaker identity extraction, enabling it to support more than four speakers.  

I also created a novel speaker swapping mechanism to purge old speakers from the speaker cache to make room for new ones.

It will be added to FluidAudio after some adjustments.

See the [Python conversion script](https://github.com/Audivize-AI/Streaming-Sortformer-Conversion).
