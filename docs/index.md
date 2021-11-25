### Abstract

One of the trademarks of reinforcement learning is that the learning agent is in an experimental setting. The agent is able to perform a certain set of actions in the envrionment given specific states and observe the outcome of the chosen action. Deep learning has contributed to the ability to solve complex problems in the reinforcement learning domain thanks to the high functional capacity of neural networks. However, generalization proves difficult as the fitted models are too fragile when the task becomes more complex and the environments are changed, that is distributional shift of the inputs occurr. A rather natural approach generalization is to learn representation of objects that represent core knwoledge about the task in the environment.  I propose that learning reusable objectsis possible by systematically exploring i.e.  experimenting in the environment with an appropriate inductive bias. For this, we take a brief look at category theory reasoning about how one can describe abstract objects from a functional perspective. That is, to represent objects by the transformations they elicit. In this sense, exploring the environment systematically is an important component for finding objects.This leads to defining an appropriate inductive bias based on cycles that characterizes the impact state-action pair has on the agents curiosity. The objects in the environment are characterized by a unique action
inherent to the object together with a raw partial view, where the agent sees only what’s immediately in front. Here, the raw view only acts as
a key for finding the object. The agent is tested on a series of MiniGrid Key-Door environments where the agent must open doors with a key and
pickup certain objects. A reward is only provided when finding the goal. Overall, We perform similar to state of the art methods, with a much
simpler algorithm and requiring almost 10x less steps to achieve similar performance.

#### Structure of state-space for object and non-object views
![Book logo](/cyclophobic-reinforcement-learning/assets/cyclemapping.png)
- Given a successful trajectory that can be extracted from a smaller environment.
- For views that contain an object we have that the mappings $$f$$ and $g$ are injective i.e. one-to-one.
- For views that don't contain an object the mappings $f$ and $g$ are surjective.  
- In some cases themappings may also be surjective for views with objects, however the number of elements in $A$ and $B $ is considerablys maller than in the non-object case.
