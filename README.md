## Overview about Neuroevolution of Augmenting Topologies (*NEAT*) ##

Advisor: Michael Adam 

This repository contains my seminar paper about Neuroevolution of Augmenting 
Topologies including its LaTeX source code, which intends to give a good 
overview of the current state of research.
Until the paper is done does this README contain interesting potential research
resources.
Feedback welcome.



#### Very Important Resources ##################################################

_Open Tabs_
https://en.wikipedia.org/wiki/Evolutionary_algorithm
https://en.wikipedia.org/wiki/Genetic_algorithm
https://en.wikipedia.org/wiki/Neuroevolution
https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies
https://en.wikipedia.org/wiki/Reinforcement_learning

http://www.scholarpedia.org/article/Neuroevolution
http://www.scholarpedia.org/article/Reinforcement_learning

<Created Papers up to now>

https://eng.uber.com/deep-neuroevolution/
https://www.oreilly.com/ideas/neuroevolution-a-different-kind-of-deep-learning

--------------------------------------------------------------------------------


* [ ] Miikkulainen - Neuroeovlution
      http://www.cs.utexas.edu/users/ai-lab/pubs/miikkulainen.encyclopedia10-ne.pdf
* [ ] Stanley, Miiikkulainen - Evolving Neural Networks throughAugmenting Topologies (**2002**)
      http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
      http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
      http://nn.cs.utexas.edu/downloads/papers/stanley.phd04.pdf
* [ ] *Primer* Stanley - Webpage describing NEAT (**2003**)
      http://www.cs.ucf.edu/~kstanley/neat.html
      http://www.cs.ucf.edu/~kstanley/#publications
* [ ] *Primer* Stanley - Webpage describing HyperNEAT (**2009**)
      https://eplex.cs.ucf.edu/hyperNEATpage/
* [ ] Stanley, D'Ambrosio, et al - A Hypercube-Based Indirect Encoding for Evolving Large-Scale Neural Networks (**2009**)
      http://axon.cs.byu.edu/~dan/778/papers/NeuroEvolution/stanley3**.pdf
* [ ] Risi, Lehman, O.Stanley - Evolving the placement and density of neurons in the hyperneat substrate (**2010**)
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.365.5509
* [ ] Risi, O.Stanley - Enhancing ES-HyperNEAT to Evolve More Complex Regular Neural Networks
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.365.4332
* [ ] Miikkulainen - Tutorial III: Evolving neural networks (**2015**)
      https://ieeexplore.ieee.org/document/7317663
* [ ] *Collection* Stanley, Clune, Uber - Welcoming the Era of Deep Neuroevolution (**2017**)
      https://eng.uber.com/deep-neuroevolution/
      https://github.com/uber-research/deep-neuroevolution
      https://eng.uber.com/accelerated-neuroevolution/
* [ ] *Paper* Real, Moore, et al - Large Scale Evolution of Image Classifiers (**2017**)
      Cited by my resources: 2
      https://arxiv.org/abs/1703.01041
      > Authors employ simple evolutionary techniques at unprecedented scales to discover models for the CIFAR-10 and CIFAR-100 datasets, starting from trivial initial conditions.
      > Neuroevolution is capable of constructing large, accurate networks starting from trivial initial conditions while searching a very large space. The process described, 
        once started, needs no participation from the experimenter.
      > A large search space is enabled by allowing to insert and remove whole layers and not restricting the possible values to choose from (yielding a dense search space). 
        Backpropagation is used to optimise the weights. The schema of the used EA is an example of tournament selection: a worker selects two individuals from the population 
        at each evolutionary step and compares their fitness. The worst of the pair is immediately removed, while the best pair is selected to be a parent. A parent undergoes 
        reproduction by producing a copy of itself and applying mutation. The modified copy (a child) is trained and put back into the population. The worker picks a mutation 
        at random from a predetermined set (altering learning rate, reset weights, insert and remove convolutions, add and skip connections, etc.).
      > "In this paper we have shown that (i) neuro-evolution is capable of constructing large, accurate networks for two challenging and popular image classification benchmarks; 
        (ii) neuro-evolution can do this starting from trivial initial conditions while searching a very large space; (iii) the process, once started, needs no experimenter participation; 
        and (iv) the process yields fully trained models."
* [ ] *Paper* Real, Aggarwal, et al - Regularized Evolution for Image Classifier Architecture Search
      https://arxiv.org/pdf/1802.01548.pdf
* [ ] *Paper* Miikkulaien, Liang, et al - Evolving Deep Neural Networks (**2017**)
      Cited by my resources: 2
      http://nn.cs.utexas.edu/?miikkulainen:chapter18
      https://arxiv.org/pdf/1703.00548.pdf
      > by Miikkulainen (the co-author of the original NEAT paper)
      > DeepNEAT differs from NEAT in that each node in the chromosome no longer represents a neuron, but a layer in a DNN. A variant of DeepNEAT, called Coevolution DeepNEAT (CoDeepNEAT), 
        separately evolves both network structure and composing modules structure. Authors used it for CNN and LSTM networks.
      > Read again the good summary of this paper in: https://www.inovex.de/blog/neuroevolution/
      > The conclusion of the paper reads as follows: "Evolutionary optimisation makes it possible to construct more complex deep learning architectures than can be done by hand. 
        The topology, components, and hyperparameters of the architecture can all be optimised simultaneously to fit the requirements of the task, resulting in superior performance."
* [ ] *Paper* Salimans, Ho, et al - Evolution Strategies as a Scalable Alternative to Reinforcement Learning (**2017**)
      https://openai.com/blog/evolution-strategies/ 
      https://arxiv.org/abs/1703.03864
* [ ] *Paper* Sun, Xue - Evolving Deep Convolutional Neural Networks for Image Classification (**2017**)
      https://arxiv.org/abs/1710.10741
      Cited by my resources: 2
      > Read again the good summary of this paper in: https://www.inovex.de/blog/neuroevolution/
      > also called 'EvoCNN'
* [ ] Rawal, Miikkulainen - From Nodes to Networks: Evolving Recurrent Neural Networks (**2018**)
      http://nn.cs.utexas.edu/?rawal:arxiv18
* [ ] Jaderberg, Daliard, et al - Population Based Training of Neural Networks (**2018**)
      https://arxiv.org/abs/1711.09846
* [ ] Elsken, Metzen, et al - Efficient Multi-objective Neural Architecture Search via Lamarckian Evolution (**2018**)
      https://arxiv.org/abs/1804.09081
* [ ] Liu, Simonyan, et al - Hierarchical Representations for Efficient Architecture Search
      https://openreview.net/pdf?id=BJQRKzbA-
* [ ] Geard, Wiles - Structure and dynamics of a gene network model (**2013**)
      https://ieeexplore.ieee.org/document/1299575
* [ ] Reil - Dynamics of Gene Expression in an Artifical Genome
      http://users.encs.concordia.ca/~kharma/coen6321/Papers/Reil-1999.pdf
* [ ] Duerr, Mattiussi, et al - Neuroevolution with Analog Genetic Encoding
      https://infoscience.epfl.ch/record/87949/files/DuerrMattiussiFloreano2006_PPSNIX_NeuroAGE.pdf
* [0] *Primer* Neuroevolution: A Primer On Evolving Artificial Neural Networks (**2018**)
      [Difficulty: Medium, Audience: Advanced, Quality: 5/5]
      https://www.inovex.de/blog/neuroevolution/
      > References: Large Scale Evolution of Image Classifiers, Evolving Deep Neural Networks
* [ ] Reisinger, Miikkulainen - Acquiring Evolvability through Adaptive Representations
      http://nn.cs.utexas.edu/downloads/papers/reisinger.gecco07.pdf
* [ ] Floreano, Duerr, Mattiussi - Neuroevolution: From Architectures to Learning (**2008**)
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.182.1567
* [ ] Kovacs - Genetics-based Machine Learning (**2010**)
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.301.1537
* [ ] Orchard, Wang - The Evolution of a Generalized Neural Learning Rule
      https://cs.uwaterloo.ca/~jorchard/academic/OrchardLin_IJCNN16.pdf
* [ ] Hausknecht, Lehman, Miikkulainen - a neuro-evolution approach to general atari game playing (**2013**)
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.494.7479
* [ ] Yao - Evolving Artificial Neural Networks 
      http://avellano.fis.usal.es/~lalonso/compt_soft/articulos/yao99evolving.pdf
* [ ] Fernando, Banarse, et al - Convolution by Evolution: Differentiable Pattern Producing Networks (**2016**)
      https://arxiv.org/abs/1606.02580



#### Neuroevolution and Broader Resources ######################################

* [ ] *Primer* Stanley - Competitive Coevolution through Evolutionary Complexification (**2004**)
      https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume21/stanley04a-html/jairhtml.html
* [1] *Blogpost* 2017, The Year of Neuroevolution (**2017**)
      (Difficulty: Easy, Audience: Overview, Quality: 3/5)
      https://medium.com/@moocaholic/2017-the-year-of-neuroevolution-30e59ae8fe18
      > Based on: 'Evolution Strategies as a Scalable Alternative to Reinforcement Learning' (Background Knowledge)
      > Evolution Strategies (ES) can be a strong alternative to Reinforcement Learning (RL) and have a number of advantages like ease of implementation, invariance to the length 
        of the episode and settings with sparse rewards, better exploration behaviour than policy gradient methods, ease to scale in a distributed setting
      > ES scales extremely well with the number of CPUs available demonstrating linear speedups in run time
      > The communication overhead of implementing ES in a distributed setting is lower than for reinforcement learning methods such as policy gradients and Q-learning.
      > The whole history of deep learning is full of re-invention and re-surrection, the main neural network learning algorithm, the backpropagation, was reinvented several times. 
        (http://people.idsia.ch/~juergen/who-invented-backpropagation.html)
      > NEAT is a TWEANN (Topology- and Weight-Evolving Artificial Neural Networks)
      > References: Genetic CNN, Large Scale Evolution of Image Classifiers, Evolving Deep Neural Networks, NMode - Neuro-MODule Evolution, PathNet, Evolution Channels Gradient Descent 
        in Super Neural Networks
* [ ] *Primer* Stanley - Neuroevolution: A different kind of deep learning (**2017**)
      https://www.oreilly.com/ideas/neuroevolution-a-different-kind-of-deep-learning
* [1] *Blogpost* Introduction to Evolutionary Algorithms (**2018**)
      (Difficulty: Very Easy, Audience: Beginners, Quality: 3/5)
      https://towardsdatascience.com/introduction-to-evolutionary-algorithms-a8594b484ac
* [0] *Blogpost* Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning (**2018**)
      https://towardsdatascience.com/deep-neuroevolution-genetic-algorithms-are-a-competitive-alternative-for-training-deep-neural-822bfe3291f5
      > Mostly referencing and Summarizing Uber's (Deep / Accelerated - Neuroevolution) and OpenAI's (Evolution Strategies as a scalable Alternative to Reinforcement Learning) research 
      > These results indicate that GAs (and RS) are not all out better or worse than other methods of optimising DNN, but that they are a competitive alternative that one can add to their RL tool belt.
        Like OpenAI, they state that although DNNs don't struggle with local optima in supervised learning, they can still get into trouble in RL tasks due to a deceptive or sparse reward signal. 
        It is for this reason that non gradient based methods such as GAs can perform well compared to other popular algorithms in RL.
* [0] *Primer* Neuroevolution: A Primer On Evolving Artificial Neural Networks (**2018**)
      [Difficulty: Medium, Audience: Advanced, Quality: 5/5]
      https://www.inovex.de/blog/neuroevolution/
      > References: Large Scale Evolution of Image Classifiers, Evolving Deep Neural Networks



#### NEAT/HyperNEAT Resources ##################################################

* [ ] *Podcast* Stanleys Podcast about NEAT (**2018**)
      https://twimlai.com/twiml-talk-94-neuroevolution-evolving-novel-neural-network-architectures-kenneth-stanley/
* [1] *Blogpost* NEAT, An Awesome Approach to NeuroEvolution (**2019**)
      (Difficulty: Easy, Audience: Beginners, Quality: 4/5)
      https://towardsdatascience.com/neat-an-awesome-approach-to-neuroevolution-3eca5cc7930f
      > NEAT's original paper focused solely on evolving dense neural networks node by node and connection by connection
      > progress we have made with training NNs through gradient descent and back propagation need not be abandoned for a neuroevolutionary process... 
        Recent papers have even highlighted ways to use NEAT and NEAT-like algorithms to evolved neural net structure and then use 
        back propagation and gradient descent to optimize these networks
      > The NEAT algorithm chooses a direct encoding methodology because of this. Their representation is a little more complex than a simple graph 
        or binary encoding, however, it is still straightforward to understand. It simply has two lists of genes, a series of nodes and a series of connections.
      > Cites: 'Evolving Deep Neural Networks'
* [1] *Blogpost* HyperNEAT: Powerful, Indirect Neural Network Evolution (**2019**)
      (Difficulty: Easy, Audience: Beginners, Quality: 3/5)
      https://towardsdatascience.com/hyperneat-powerful-indirect-neural-network-evolution-fba5c7c43b7b
      > DNA is an indirect encoding because the phenotypic results (what we actually see) are orders of magnitude larger than the genotypic content (the genes in the DNA). 
        If you look at a human genome, we\u2019ll say it has about 30,000 genes coding for approximately 3 billion amino acids. Well, the brain has 
        3 trillion connections. Obviously, there is something indirect going on here!



#### Unsorted Wikipedia Resources ##############################################

* [ ] Neuroevolution
      https://en.wikipedia.org/wiki/Neuroevolution
* [ ] Evolutionary Algorithm 
      https://en.wikipedia.org/wiki/Evolutionary_algorithm
* [ ] Evolution Strategy
      https://en.wikipedia.org/wiki/Evolution_strategy
* [ ] Evolutionary Programming
      https://en.wikipedia.org/wiki/Evolutionary_programming
* [ ] Evolutionary Acquistion of Neural Topologies
      https://en.wikipedia.org/wiki/Evolutionary_acquisition_of_neural_topologies
* [ ] Genetic Algorithm
      https://en.wikipedia.org/wiki/Genetic_algorithm
* [ ] Genetic Programming
      https://en.wikipedia.org/wiki/Genetic_programming
* [ ] Artifical Development
      https://en.wikipedia.org/wiki/Artificial_development
* [ ] NEAT
      https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies
* [ ] HyperNEAT
      https://en.wikipedia.org/wiki/HyperNEAT
* [ ] Fitness Function
      https://en.wikipedia.org/wiki/Fitness_function
* [ ] Memetic Algorithm
      https://en.wikipedia.org/wiki/Memetic_algorithm
* [ ] General Game Playing
      https://en.wikipedia.org/wiki/General_game_playing
* [ ] Evolutionary Robotics
      https://en.wikipedia.org/wiki/Evolutionary_robotics



#### Research Paper Resources ##################################################

* [ ] Whiteson, Stone, Stanley, Miikkulainen, et al - Automatic feature selection in neuroevolution (**2005**) 
      https://dl.acm.org/citation.cfm?id=1068210
* [ ] Stanley, Bryant, Miikkulainen - Real-time Neuroevolution in the NERO Video Game (**2005**)
      http://nn.cs.utexas.edu/downloads/papers/stanley.ieeetec05.pdf
* [ ] Gomez, Schmidhuber, Miikkulainen - Efficient Non-linear Control Through Neuroevolution (**2006**)
      https://link.springer.com/chapter/10.1007/11871842_64
* [ ] Chen, Alahakoon - NeuroEvolution of Augmenting Topologies with Learning for Data Classification (**2007**)
      https://www.researchgate.net/publication/4255903
* [ ] Stanley, D'Ambrosio, et al - A Hypercube-Based Indirect Encoding for Evolving Large-Scale Neural Networks (**2009**)
      http://axon.cs.byu.edu/~dan/778/papers/NeuroEvolution/stanley3**.pdf
* [ ] Buk, Snorek, et al - NEAT in HyperNEAT substituted with genetic programming (**2009**)
      https://www.researchgate.net/publication/225720114
* [ ] Cardamone, Loiacono - Learning to Drive in the Open Racing Car Simulator Using Online Neuroevolution (**2010**)
      https://ieeexplore.ieee.org/document/5482132
* [ ] Cheung, Sable, et al - Hybrid Evolution of Convolutional Networks (**2011**)
      https://ieeexplore.ieee.org/document/6146987
* [ ] Lowell, Birger, et al - Comparison of NEAT and HyperNEAT on a Strategic Decision-Making Problem (**2011**) 
      http://web.mit.edu/jessiehl/Public/aaai11/fullpaper.pdf
* [ ] Cuccu, Luciw, et al - Intrinsically motivated neuroevolution for vision-based reinforcement learning (**2011**)
      https://ieeexplore.ieee.org/document/6037324
* [ ] Lowell, Grabkovsky - Comparison of NEAT and HyperNEAT Performance on a Strategic Decision-Making Problem (**2011**)
      https://ieeexplore.ieee.org/document/6042728
* [ ] Pereira, Petry - Data Assimilation using NeuroEvolution of Augmenting Topologies (**2012**) 
      https://www.researchgate.net/publication/237049328
* [ ] Risi, Stanley - An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density and Connectivity of Neurons (**2012**)
      Cited by my resources: 1
      https://eplex.cs.ucf.edu/papers/risi_alife12.pdf
* [ ] Schrum, Miikkulainen - Evolving Multimodal Networks for Multitask Games (**2012**)
      https://ieeexplore.ieee.org/document/6179519
* [ ] Moriguchi, Honiden - CMA-TWEANN: Efficient Optimization of Neural Networks via Self-Adaptation and Seamless Augmentation (**2012**)
      https://www.researchgate.net/publication/261851261
* [ ] Gallego-Dur'an, Molina-Carmona - Experiments on Neuroevolution and OnlineWeight Adaptation in Complex Environments (**2013**) 
      https://core.ac.uk/download/pdf/19336609.pdf
* [ ] Verbancsics, Harguess - Generative NeuroEvolution for Deep Learning (**2013**)
      https://arxiv.org/abs/1312.5355
* [ ] Sohangir, Rahimi, et al - Optimized feature selection using NeuroEvolution of Augmenting Topologies (NEAT) (**2013**)
      https://ieeexplore.ieee.org/document/6608379
* [ ] Pugh, Stanley - Evolving Multimodal Controllers with HyperNEAT (**2013**)
      http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.867&rep=rep1&type=pdf
* [ ] Risi, Tugelius - Neuroevolution in Games: State of the Art and Open Challenges (**2014**)
      https://arxiv.org/abs/1410.7326
* [ ] Caamano, Bellas, et al - Augmenting the NEAT algorithm to improve its temporal processing capabilities (**2014**)
      https://ieeexplore.ieee.org/document/6889488
* [ ] Huang, Lehman, Miikkulainen - Grasping novel objects with a dexterous robotic hand through neuroevolution (**2014**)
      https://ieeexplore.ieee.org/document/7013242
* [ ] Miikkulainen - Tutorial III: Evolving neural networks (**2015**)
      https://ieeexplore.ieee.org/document/7317663
* [ ] Verbancsics, Harguess - Image Classification Using Generative Neuro Evolution for Deep Learning (**2015**)
      https://ieeexplore.ieee.org/abstract/document/7045925
* [ ] Samothrakis, Perez-Liebana - Neuroevolution for General Video Game Playing (**2015**)
      https://ieeexplore.ieee.org/document/7317943
* [ ] Cussat-Blanc, Harrington, et al - Gene Regulatory Network Evolution Through Augmenting Topologies (**2015**)
      https://ieeexplore.ieee.org/abstract/document/7018989
* [ ] Lample, Chaplot - Playing FPS Games with Deep Reinforcement Learning (**2016**)
      https://arxiv.org/abs/1609.05521
* [ ] Kristo, Maulidevi - Deduction of fighting game countermeasures using Neuroevolution of Augmenting Topologies (**2016**)
      https://ieeexplore.ieee.org/document/7936127
* [ ] Sorensen, Olsen, et al - Breeding a diversity of Super Mario behaviors through interactive evolution (**2016**)
      https://ieeexplore.ieee.org/document/7860436
* [ ] Boris, Goran - Evolving neural network to play game 2048 (**2016**)
      https://ieeexplore.ieee.org/document/7818911
* [ ] Rawal, Miikkulainen - Evolving Deep LSTM-based Memory networks using an Information Maximization Objective (**2016**)
      http://nn.cs.utexas.edu/?rawal:gecco2016
* [ ] Braylan, Miikkulainen - Reuse of Neural Modules for General Video Game Playing (**2016**)
      http://nn.cs.utexas.edu/?braylan:aaai16
* [ ] Rodzin, Rodzina - Neuroevolution: Problems, algorithms, and experiments (**2017**)
      https://ieeexplore.ieee.org/document/7991745
* [ ] Xie, Yuille - Genetic CNN (**2017**)
      https://arxiv.org/abs/1703.01513
      > explores the idea of learning deep network structures automatically. Authors proposed a genetic algorithm to create new network structures.
* [ ] Rawal, Miikkulainen - From Nodes to Networks: Evolving Recurrent Neural Networks (**2018**)
      http://nn.cs.utexas.edu/?rawal:arxiv18
* [ ] Ghazi-Zahedi - NMode - Neuro-MODule Evolution (**2017**)
      https://arxiv.org/abs/1701.05121
      > Authors showed that NMODE was able to evolve a locomotion behaviour for a standard six-legged walking machine in approximately 10 generations and showed how it can be used 
        for incremental evolution of a complex walking machine.
* [ ] Fernando, Banarse, et al - PathNet, Evolution Channels Gradient Descent in Super Neural Networks (**2017**) 
      https://arxiv.org/abs/1701.08734
      > PathNet is a neural network algorithm that uses agents embedded in the neural network whose task is to discover which parts of the network to re-use for new tasks. 
      > Agents are pathways (views) through the network which determine the subset of parameters that are used and updated by the forwards and backwards passes of the 
        backpropogation algorithm. During learning, a tournament selection genetic algorithm is used to select pathways through the neural network for replication and mutation. 
        Authors demonstrate successful transfer learning; fixing the parameters along a path learned on task A and re-evolving a new population of paths for task B, allows task B 
        to be learned faster than it could be learned from scratch or after fine-tuning. Paths evolved on task B re-use parts of the optimal path evolved on task A. 
        Positive transfer was demonstrated for binary MNIST, CIFAR, and SVHN supervised learning classification tasks, and a set of Atari and Labyrinth reinforcement learning tasks, 
        suggesting PathNets have general applicability for neural network training.
* [ ] Ferner, Fischler, et al - Combining Neuro-Evolution of Augmenting Topologies with Convolutional Neural Networks (**2017**)
      https://www.researchgate.net/publication/328939814
* [ ] Wen, Guo, et al - Neuroevolution of augmenting topologies based musculor-skeletal arm neurocontroller (**2017**) 
      https://ieeexplore.ieee.org/document/7969727
* [ ] Pastorek, O'Neill - Historical Markings in Neuroevolution of Augmenting Topologies Revisited (**2017**) 
      https://www.researchgate.net/publication/321148165
* [ ] Desell - Developing a Volunteer Computing Project to Evolve Convolutional Neural Networks and Their Hyperparameters (**2017**)
      https://ieeexplore.ieee.org/document/8109119
* [ ] Alvernaz, Togelius - Autoencoder-augmented neuroevolution for visual doom playing (**2017**)
      https://ieeexplore.ieee.org/abstract/document/8080408
* [ ] Assuncao, Lourenco, et al - Towards the Evolution of Multi-Layered Neural Networks: A Dynamic Structured Grammatical Evolution Approach (**2017**)
      https://arxiv.org/abs/1706.08493
* [ ] Such, Madhave, Stanley - Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning (**2017**)
      https://www.researchgate.net/publication/321902574
* [ ] Wen, Guo, et al - Neuroevolution of augmenting topologies based musculor-skeletal arm neurocontroller (**2017**)
      https://ieeexplore.ieee.org/document/7969727
* [ ] Kadish - Clustering sensory inputs using NeuroEvolution of Augmenting Topologies (**2018**) 
      https://www.researchgate.net/publication/325877035
* [ ] Aly, Dugan - Experiential Robot Learning with Accelerated Neuroevolution (**2018**)
      https://arxiv.org/abs/1808.05525
* [ ] Lehman, Clune, Stanley, et al - Safe mutations for deep and recurrent neural networks through output gradients (**2018**)
      https://www.researchgate.net/publication/326163251
* [ ] Ha - Neuroevolution for deep reinforcement learning problems (**2018**)
      https://www.researchgate.net/publication/326238939
* [ ] Iba - Evolutionary Approach to Deep Learning (**2018**)
      https://www.researchgate.net/publication/325799313
* [ ] Omelianenko - Neuroevolution - evolving Artificial Neural Networks topology from the scratch (**2018**)
      https://www.researchgate.net/publication/325848235
* [ ] Baldominos, Saez - Evolutionary Convolutional Neural Networks: an Application to Handwriting Recognition (**2018**)
      https://www.researchgate.net/publication/322079459
* [ ] Wang, Clune, Stanley - VINE: An Open Source Interactive Data Visualization Tool for Neuroevolution (**2018**)
      https://arxiv.org/abs/1805.01141
* [ ] Baldominos, Saez - On the Automated, Evolutionary Design of Neural Networks-Past, Present, and Future (**2019**)
      https://www.researchgate.net/publication/331864343
* [ ] Vargas, Murata - Spectrum-Diverse Neuroevolution with Unified Neural Models (**2019**)
      https://arxiv.org/abs/1902.06703
* [ ] Stanley, Miikkulainen, et al - Designing neural networks through neuroevolution (**2019**)
      https://www.nature.com/articles/s42256-018-0006-z
* [ ] Behjat, Chidambaran, et al - Adaptive Genomic Evolution of Neural Network Topologies (AGENT) for State-to-Action Mapping in Autonomous Agents (**2019**)
      https://arxiv.org/abs/1903.07107
* [ ] Aly, Weikersdorfer, et al - Optimizing Deep Neural Networks with Multiple Search Neuroevolution (**2019**)
      https://arxiv.org/abs/1901.05988
* [ ] Assuncao, Lourenco, et al - Fast DENSER: Efficient Deep NeuroEvolution (**2019**)
      https://www.researchgate.net/publication/332306893
* [ ] Liang, Miikkulainen, et al - Evolutionary Neural AutoML for Deep Learning (**2019**)
      http://nn.cs.utexas.edu/?liang:gecco19
* [ ] Miller, Cussat-Blanc - Evolving Programs to Build Artificial Neural Networks (**2020**)
      https://www.researchgate.net/publication/332470796



#### Unsorted Implementation Resources #########################################

* [ ] Collection of most of the published NEAT software
      https://eplex.cs.ucf.edu/neat_software/
* [ ] NEAT Visualizer SFML (**2014**)
      https://sourceforge.net/projects/neatvisualizers/
* [ ] MarI/O - Machine Learning for Videogames (NEAT) (**2015**)
      https://www.youtube.com/watch?v=qv6UVOQ0F44
* [ ] NEAT-Python (**2018**)
      https://neat-python.readthedocs.io/en/latest/
      https://github.com/CodeReclaimers/neat-python 
* [ ] PyTorch NEAT's adaptive HyperNEAT (**2018**)
      https://github.com/uber-research/PyTorch-NEAT
* [ ] TensorflowNEAT (**2019**)  
      https://github.com/PaulPauls/TensorFlow-NEAT
      https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2018_2019/presentation/S8/MP_Cristian.pdf
* [ ] DeepLearning on Car Simulator (**2018**)
      https://towardsdatascience.com/deep-learning-on-car-simulator-ff5d105744aa
* [ ] Self Driving Car Simulation Unity 3D using Genetic Algorithms and Neural Networks (**2019**)
      https://www.youtube.com/watch?v=m8fYPy9eiOo
      https://github.com/dDevTech/Self-Driving-Car-Tutorial



#### Background Knowledge Resources ############################################

* [ ] Tensorflow Eager for required dynamic computation graphs: 
      https://www.tensorflow.org/guide/eager
* [ ] Intro to Reinforcement-Learning: 
      https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419
* [ ] Intro to Reinforcement-Learning: 
      https://www.geeksforgeeks.org/what-is-reinforcement-learning/
* [ ] Beginners Guide to Deep Reinforcement Learning: 
      https://skymind.ai/wiki/deep-reinforcement-learning
* [ ] Wikipedia: Artifical Neural Network
      https://en.wikipedia.org/wiki/Artificial_neural_network



#### Other/Unsorted Resources ##################################################

* Neuroevolution Research Group of UTexas (Stanley, Miikkulainen)
  http://nn.cs.utexas.edu/?neuroevolution
* https://github.com/openai/evolution-strategies-starter
* http://nn.cs.utexas.edu/?neuroevolution-tutorial-ijcnn2013
* http://nerogame.org/
* http://www.cs.utexas.edu/users/nn/keyword?rtneat
* https://github.com/sean-dougherty/accneat
* http://gekkoquant.com/2016/03/13/evolving-neural-networks-through-augmenting-topologies-part-1-of-4/
* https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT
* http://gar.eecs.ucf.edu/
* http://groups.yahoo.com/group/neat/
* https://www.youtube.com/watch?v=Tyhbf0vgwP0
* https://www.youtube.com/watch?v=WyDMbfCARW8

