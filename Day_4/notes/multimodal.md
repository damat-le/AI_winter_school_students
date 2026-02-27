# Multimodal models

Lecturer: Lorenzo Baraldi


## CLIP

* CLIP: 
    * text encoder
    * image encoder
    * during training, we build a matrix of similarities between all the text and all the images, and we train the model to maximize the similarity of the correct pairs and minimize the similarity of the wrong pairs. (slide 9)
    * loss: contrastive loss (slide 10)
    * killer feature: zero-shot classification (slide 12)


* CLIP variants: 
    * see slide 20
    * one notable variant is the sigmoid loss variant, which is more efficient for parallel training because it does not requires communication between the different GPUs (slide 20). In fact, it is like a pointwise loss. 


* One problem with CLIP:
    * CLIP has limited use cases, since it is only useful for connecting images and text embeddings.


* One way to finetune CLIP is with learnable prompts (slide 26). This is a very efficient way to finetune the model, since it only requires to learn a few parameters (the prompt) instead of the whole model.


According to the lecturer, contrastive learning is nowadays the most popular approach to implement the visual backbone of multimodal models.


* Open source datasets:
    * Today there are many of them
    * LAION-5B (slide 16) is one resulting from a european effort.

## Multimodal models that do not have separate encoders for each modality

* Visual BERT:
    * it is a transformer that takes as input both the image and the text, and it learns to attend to both modalities at the same time. 
    * wrt clip, visual bert is early fusion, while clip is late fusion.

Assume you what to do multimodal retrieval: 
    * you have a query (text or image) and you want to retrieve the most relevant images or texts.
    * with clip, you can encode the query and the database separately, and then compute the similarity between the query and the database. 
    * with visual bert, you have to encode the query and the database together, which is more computationally expensive.
    * in literature, there are many variants that combine early and late fusion, to get the best of both worlds.


* FLAVA:
    * hybrid architecture: separate encoders for each modality, but also a joint encoder that takes as input both the image and the text.
    * it is trained with multiple losses: visual loss (masking), text loss (masking), and a multimodal loss that tries to align the visual and textual representations.
    * see slide 38

## Encoder-Decoder models

Up to now, we have seen encoder only approaches.

* SimVLM:
    * it is an encoder-decoder transformer
    * the encoder takes as input the image and the initial part of the text, and the decoder then is in charge to  generate the rest of the text.

* CoCa:
    * interesting because it can both do similarity and generation.
    * you have a image encoder
    * you have a unimodal text dencoder (it is technically a decoder, but it is used as an encoder for the similarity task) that takes the input text.
    * you have multimodal decoder that takes as input both the image and the text, and it is used for generation, specifically it generates a caption for the image (slide 40).
    * More details:
        * the image encoder produces two outputs: fine grained emb and also a single global embedding that acts like a summarization of the image (created by attentional pooling). 
        * the summary embedding is used for the similarity task
        * while the fine grained emb is used for the generation task.
        * the summary vector is used to compute a contrastive loss with the unimodal text encoder.
        * the unimodal text encoder is used for the similarity task, while the multimodal decoder is used for the generation task.

In general, when you have to summarize info in a single vector, attentional pooling can be useful. Keep it in mind.

---

How to adapt a pretrained LLM to be multimodal? 

* FROZEN - 2021 (slide 44): 
    * you have a COMPLETELY frozen LLM and you add some adapters to it, that are trained to process the visual information and to inject it into the frozen LLM.
    * vision encoder is trainable 
    * this is the first model that demonstrated that you can get good performance on multimodal tasks by just adding a few adapters to a frozen LLM, without the need to finetune the whole model.
    * in terms of performance, it is not a way to create a SOTA modal but the concept is very interesting because it means that an adapter can create embeddings that are compatible with those of the frozen LLM.


* FLAMINGO:
    * one of the first truly multimodal LLM
    * you have a frozen LLM backbone
    * It is able to encode both images and videos into a fixed length sequence of vectors (slide 46): 
        * to do that: perceiver resampler (similar to the attention pooling)
        * idea of perveicer resampler: is a transformer that takes as input the visual features (image of video frames) and produces a FIXED number of output vectors that are then fed to the LLM.
        * only a portion of the input (learnable latent queries) is used to create the query in the attention block
        * the visual features, together with the latent queries, are used to create the key and value in the attention block.
        * the length of the output corresponds to the number of latent queries, which is a hyperparameter that can be tuned (fixed).
        * So this module takes as input a variable number of visual features and produces a fixed number of output vectors that are then fed to the LLM.
    * How to combine the visual and textual information in the LLM? Gated XATTN
        * before every two transformer block of the LLM, a new cross-attention layer is inserted.
        * the cross attention layer takes as input the output of the perceiver resampler and the output of the previous transformer block, and it produces a new representation that is then fed to the next transformer block.
        * the cross attention layer is gated, meaning that  at the beginning the contribution of this layer is set to zero and then is gradually increased during training.
        * this gradual gating mechanism (besed on tanh) prevents the LLM to collapse and start generating random text.
        * this layer is thus called gated cross attention
        * note: the visual features are injected at every gate cross attention layers, so the visual information enter the global architecture in several points, like refreshing the visual knowlenge also in deep layes.

* BLIP / BLIP - 2023 (slide 49):
    * a way to connect a frozen LLM with a frozen vision encoder. 
    * Two versions: 
        1. boostrapping from decoder-based LLM (like gpt)
        2. boostrapping from encoder-decode-based LLM (like FlanT5)

How to create SOTA multimodal models?

* LaVA - 2023:
    * it can encode visual info
    * uses instruction following to align the visual and textual representations in the same space.

    * Architecture: 
        * you have a almost frozen LLM
        * you have a frozen vision encoder and a projection layer on top of it that is trained to align the visual and textual representations in the same space.
        * 
    * Training:
        * the training is done in two stages:
            1. first stage: you train the projection layer to align the visual and textual representations in the same space (Vision and text encoders are frozen).
            2. second stage: you finetune the LLM model (visual encoder frozen) on a multimodal instruction following dataset (slide 65).
    * How to get the data? 
        * the passed images to a text only LLM by passing through a image captioner model. In this way the text-only LLM is able to "see" the image through the caption.
        * using a object detector to extract the objects from the image, you tell the text-only model what are the objects in the image and where they are located. In this way the text-only LLM is able to "see" the image through the objects and their locations.
        * Then you ask the text-only LLM to generate a dialog about this visual scene. The dialog can also be complex and involve reasoning.
        * the resulting dialog, together with the original image, can be used as training data for a multimodal model.
        * Visual Chat dataset is created in this way (slide 66).


Computer vision challenge: 
- agentic multimodal model, calling tools.