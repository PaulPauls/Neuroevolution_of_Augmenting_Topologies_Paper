## Overview about Neuroevolution of Augmenting Topologies (*NEAT*) ##

Advisor: Michael Adam (Technical University Munich)

This repository contains my seminar paper about Neuroevolution of Augmenting 
Topologies including its LaTeX source code, which intends to give a good 
overview of the current state of research.
Until the paper is done does this README contain interesting potential research
resources.
Feedback welcome.



#### Unsorted Wikipedia Resources ####

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



#### NEAT/HyperNEAT Resources ####

* [ ] *Primer* Stanley - Webpage describing NEAT (**2003**)
      http://www.cs.ucf.edu/~kstanley/neat.html
* [ ] *Primer* Stanley - Webpage describing HyperNEAT (**2009**)
      https://eplex.cs.ucf.edu/hyperNEATpage/
* [ ] *Bookchapter* Automatic Task Decomposition for the NeuroEvolution of Augmenting Topologies (NEAT) Algorithm (**2012**)
      https://link.springer.com/chapter/10.1007/978-3-642-29066-4_1
* [ ] *Bookchapter* HyperNEAT - The First 5 years (**2014**)
      https://link.springer.com/chapter/10.1007/978-3-642-55337-0_5
* [ ] *Podcast* Stanleys Podcast about NEAT (**2018**)
      https://twimlai.com/twiml-talk-94-neuroevolution-evolving-novel-neural-network-architectures-kenneth-stanley/
* [ ] *Video Lecture* NEAT/HyperNEAT in evolutionary robotics (**2018**)
      https://www.youtube.com/watch?v=MZzJ-EB-_yA
* [1] *Blogpost* NEAT, An Awesome Approach to NeuroEvolution (**2019**)
      (Difficulty: Easy, Audience: Beginners)
      https://towardsdatascience.com/neat-an-awesome-approach-to-neuroevolution-3eca5cc7930f
      > NEAT's original paper focused solely on evolving dense neural networks node by node and connection by connection
      > progress we have made with training NNs through gradient descent and back propagation need not be abandoned for a neuroevolutionary process... 
        Recent papers have even highlighted ways to use NEAT and NEAT-like algorithms to evolved neural net structure and then use 
        back propagation and gradient descent to optimize these networks
      > The NEAT algorithm chooses a direct encoding methodology because of this. Their representation is a little more complex than a simple graph 
        or binary encoding, however, it is still straightforward to understand. It simply has two lists of genes, a series of nodes and a series of connections.
      > Cites: 'Evolving Deep Neural Networks'
* [ ] *Blogpost* HyperNEAT: Powerful, Indirect Neural Network Evolution (**2019**)
      https://towardsdatascience.com/hyperneat-powerful-indirect-neural-network-evolution-fba5c7c43b7b



#### Neuroevolution and Broader Resources ####

* [ ] *Primer* Stanley - Competitive Coevolution through Evolutionary Complexification (**2004**)
      https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume21/stanley04a-html/jairhtml.html
* [1] *Blogpost* Real-time Neuroevolution of Augmented Topologies in Video Games (**2008**)
      http://aigamedev.com/open/review/neuro-evolution-augmented-topologies/
      > Based on 'Real-time Neuroevolution in the NERO Video Game' (see Conceptual Resources)
      > Check out rtNEAT (real-time NEAT), possibly in the NERO framework
      > Efficient technique, it starts out with a very simple set of behavior and only expands the search space when it is found to be beneficial
      > Difference to other genetic algorithms is that it not only changes the weights of the neural net but also its structure
* [1] *Blogpost* 2017, The Year of Neuroevolution (**2017**)
      https://medium.com/@moocaholic/2017-the-year-of-neuroevolution-30e59ae8fe18
      > Based on: 'Evolution Strategies as a Scalable Alternative to Reinforcement Learning' (Background Knowledge)
      > Evolution Strategies (ES) can be a strong alternative to Reinforcement Learning (RL) and have a number of advantages like ease of implementation, invariance to the length 
        of the episode and settings with sparse rewards, better exploration behaviour than policy gradient methods, ease to scale in a distributed setting
      > ES scales extremely well with the number of CPUs available demonstrating linear speedups in run time
      > The communication overhead of implementing ES in a distributed setting is lower than for reinforcement learning methods such as policy gradients and Q-learning.
      > The whole history of deep learning is full of re-invention and re-surrection, the main neural network learning algorithm, the backpropagation, was reinvented several times. 
        (http://people.idsia.ch/~juergen/who-invented-backpropagation.html)
      > NEAT is a TWEANN (Topology- and Weight-Evolving Artificial Neural Networks)
      > Referenced: Genetic CNN, Large Scale Evolution of Image Classifiers, Evolving Deep Neural Networks, NMode - Neuro-MODule Evolution, PathNet, Evolution Channels Gradient Descent 
        in Super Neural Networks
* [ ] *Primer* Stanley - Neuroevolution: A different kind of deep learning (**2017**)
      https://www.oreilly.com/ideas/neuroevolution-a-different-kind-of-deep-learning
* [ ] *Blogpost* Introduction to Evolutionary Algorithms (**2018**)
      https://towardsdatascience.com/introduction-to-evolutionary-algorithms-a8594b484ac
* [ ] *Primer* Neuroevolution: A Primer On Evolving Artificial Neural Networks (**2018**)
      https://www.inovex.de/blog/neuroevolution/
* [ ] *Blogpost* Neuroevolution\u200a\u2014\u200aevolving Artificial Neural Networks topology from the scratch (**2018**)
      https://becominghuman.ai/neuroevolution-evolving-artificial-neural-networks-topology-from-the-scratch-d1ebc5540d84
* [ ] *Blogpost* Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning (**2018**)
      https://towardsdatascience.com/deep-neuroevolution-genetic-algorithms-are-a-competitive-alternative-for-training-deep-neural-822bfe3291f5
* [ ] *Video Lecture* Topology and Weight Evolving Artificial Neural Network (TWEANN) (**2019**)
      http://primo.ai/index.php?title=Topology_and_Weight_Evolving_Artificial_Neural_Network_(TWEANN)
* [ ] *Collection* Neuroevolution Collection
      http://nn.cs.utexas.edu/?neuroevolution
* [ ] *Primer* Beginners Guide to Genetic and Evolutionary Algorithms
      https://skymind.ai/wiki/evolutionary-genetic-algorithm



#### Research Paper Resources ####

* [ ] Stanley, Miiikkulainen - Evolving Neural Networks throughAugmenting Topologies (**2002**)
      http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
* [ ] Whiteson, Stone, Stanley, Miikkulainen, et al - Automatic feature selection in neuroevolution (**2005**) 
      https://dl.acm.org/citation.cfm?id=1068210
* [ ] Stanley, Bryant, Miikkulainen - Real-time Neuroevolution in the NERO Video Game (**2005**)
      http://nn.cs.utexas.edu/downloads/papers/stanley.ieeetec05.pdf
* [ ] Gomez, Schmidhuber, Miikkulainen - Efficient Non-linear Control Through Neuroevolution (**2006**)
      https://link.springer.com/chapter/10.1007/11871842_64
* [ ] Chen, Alahakoon - NeuroEvolution of Augmenting Topologies with Learning for Data Classification (**2007**)
      https://www.researchgate.net/publication/4255903
* [ ] Stanley, D'Ambrosio, et al - A Hypercube-Based Indirect Encoding for Evolving Large-Scale Neural Networks (**2009**)
      https://ieeexplore.ieee.org/document/6792316
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
* [ ] Risi, Stanley - An Enhanced Hypercube-Based Encoding for Evolving thePlacement, Density and Connectivity of Neurons (**2012**)
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
* [ ] Rodzin, Rodzina - Neuroevolution: Problems, algorithms, and experiments (**2017**)
      https://ieeexplore.ieee.org/document/7991745
* [ ] Xie, Yuille - Genetic CNN (**2017**)
      https://arxiv.org/abs/1703.01513
      > explores the idea of learning deep network structures automatically. Authors proposed a genetic algorithm to create new network structures.
* [ ] Real, Moore, et al - Large Scale Evolution of Image Classifiers (**2017**)
      https://arxiv.org/abs/1703.01041
      > Authors employ simple evolutionary techniques at unprecedented scales to discover models for the CIFAR-10 and CIFAR-100 datasets, starting from trivial initial conditions.
      > Neuroevolution is capable of constructing large, accurate networks starting from trivial initial conditions while searching a very large space. The process described, 
        once started, needs no participation from the experimenter.
* [ ] Miikkulaien, Liang, et al - Evolving Deep Neural Networks (**2017**)
      Cited by my resources: 2
      https://arxiv.org/abs/1703.00548
      > by Miikkulainen (the co-author of the original NEAT paper)
      > DeepNEAT differs from NEAT in that each node in the chromosome no longer represents a neuron, but a layer in a DNN. A variant of DeepNEAT, called Coevolution DeepNEAT (CoDeepNEAT), 
        separately evolves both network structure and composing modules structure. Authors used it for CNN and LSTM networks.
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
* [ ] Salimans, Ho, et al - Evolution Strategies as a Scalable Alternative to Reinforcement Learning (**2017**)
      https://openai.com/blog/evolution-strategies/ 
      https://arxiv.org/abs/1703.03864
* [ ] Assuncao, Lourenco, et al - Towards the Evolution of Multi-Layered Neural Networks: A Dynamic Structured Grammatical Evolution Approach (**2017**)
      https://arxiv.org/abs/1706.08493
* [ ] Such, Madhave - Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning (**2017**)
      https://www.researchgate.net/publication/321902574
* [ ] Miikkulainen, Liang, et al - Evolving Deep Neural Networks (**2017**)
      https://arxiv.org/abs/1703.00548
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
* [ ] Miller, Cussat-Blanc - Evolving Programs to Build Artificial Neural Networks (**2020**)
      https://www.researchgate.net/publication/332470796



#### Unsorted Implementation Resources ####

* [ ] MarI/O - Machine Learning for Videogames (NEAT) (**2015**)
      https://www.youtube.com/watch?v=qv6UVOQ0F44
* [ ] NEAT-Python (**2018**)
      https://neat-python.readthedocs.io/en/latest/
      https://github.com/CodeReclaimers/neat-python
* [ ] MultiNeat, portable neuroevolution software libray (**2018**)
      http://www.multineat.com/
      https://github.com/peter-ch/MultiNEAT
* [ ] PyTorch NEAT's adaptive HyperNEAT (**2018**)
      https://github.com/uber-research/PyTorch-NEAT
* [ ] Go implementaiton of evolvable substrate HyperNEAT (**2019**)
      https://github.com/yaricom/goESHyperNEAT
* [ ] TensorflowNEAT (**2019**)  
      https://github.com/PaulPauls/TensorFlow-NEAT
      https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2018_2019/presentation/S8/MP_Cristian.pdf
* [ ] goNEAT (**2019**)
      https://github.com/yaricom/goNEAT
* [ ] DeepLearning on Car Simulator (**2018**)
      https://towardsdatascience.com/deep-learning-on-car-simulator-ff5d105744aa
* [ ] Self Driving Car Simulation Unity 3D using Genetic Algorithms and Neural Networks (**2019**)
      https://www.youtube.com/watch?v=m8fYPy9eiOo
      https://github.com/dDevTech/Self-Driving-Car-Tutorial



#### Background Knowledge Resources ####

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



