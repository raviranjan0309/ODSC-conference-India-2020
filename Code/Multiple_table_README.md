# Reinforcement learning

![Image](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg)

Reinforcement learning (RL) is an area of machine learning concerned with how an agents ought to take actions in an environment so as to maximize some notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.For any problem to solved using reinforcement learning,we need to define following things-

  - State
  - Action
  - Agent
  - Reward/Penalty
  - Environment
  - Policy
##### State
 A state is a concrete and immediate situation in which the agent finds itself. In recommendation system, state is the previous history of the user
##### Action
In recommendation system, action is the items we are recommending to a user
##### Agent
Agent is our recommendation system
##### Reward/Penalty
Reward/Penalty is the feedback which the agent gets after taking an action in a certain state
##### Environment
For every RL problem to be coded, you need to code your environment, It basically consist the action state and next_state after that action relation. In a  state, agent recommends an item and if that item is clicked then user state changes. That is what envirnment consists of.state-action-feedback-next_state
We have designed our own environment for a Recommendation system, how  will state change from one state to other after taking action. **[OpenAI gym](https://gym.openai.com/)**   library provides environment for various problems. We can also edit those environment according to our problem, but it's mostly related to Robotics, gaming. For our problem formulation I wrote some functions to make our own environment.

##### Policy (π)
The policy is the strategy that the agent employs to determine the next action based on the current state. It maps states to actions, the actions that promise the highest reward.
### Epsilon-greedy Q-learning Algorithm
The very basic algorithm of reinforcement learning is Q learning. As reinforcement learning has property of explore and exploit, we introduce **epsilon greedy** exploration method so it's Epsilon greedy Q learning. In this algorithm,for state-action mapping as per the policy, we maintain a Q table. **Rows**-*State*,**Columns**-*Action*. Cells contain the Q value of the state action pairs.For further one can get to know from the [link](https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/) provided.
| State | Action1 | Action2 | .....Action-N|
| ------ | ------ |-------| -----|
| State1 | Q-value | Q-value | Q-value |
| State2 | Q-value | Q-value | Q-value |
| State-N  | Q-value | Q-value | Q-value |

What is Q value?
![image](https://cdn-media-1.freecodecamp.org/images/s39aVodqNAKMTcwuMFlyPSy76kzAmU5idMzk)
How to calculate Q values?
Answer- As can be seen in the figure below
![Image](https://developer.ibm.com/developer/articles/cc-reinforcement-learning-train-software-agent/images/fig03.png)

##### Challenges
As we can see, if we go for Q learning algorithm, we need to maintain Q table for action state mapping.In real scenarion, we have thousands of items to recommend and if we take previous three history of user as state then, nC3 then there will be huge number of state which can't be maintained in one table.
##### Solution

To deal with this problem, I decided to make a multiple table according to it's genre.The number of table is equal to number of genres in the data. As every article has it's genre, We take the article of same genre and forms it's combination in its table respectively for all genres. But still we will be having lots of states if we follow **nC3**, so instead of that, I stored **one** and **nC2** of it.We take state as three previous history, find the genre of each history article and then look into that table.This approach has not been followed by anyone till now, this is our approach which we found for scaling the Q learning algorithm.
##### Example-
**State=[article1, article2,article3]**
After this we form table using state .Each row of the table is a sub state of the State .
**table--**
[article1, article2,genre4, State_Space4,Q_table4,23]
[article3,genre2, State_Space2,Q_table2,3]
This is done by  ***PREPROCESS*** function which takes input (State) explained later on in the code .

This helps in easily accessing all the related information of the **sub state** very easily. Now we know the sub states and it's related information, our next step is to recommend according to the sub state using Q learning algorithm.Here the algorithm decides whether to explore or exploit according to epsilon value and maximum Q value of that state.

As we have multiple sub state of a single state, we need to append the recommended item of each sub state and provide a list of recommendation. This is done by ***RECOMMENDATION_LIST*** function in the code explained later. It takes input as State, calls ***PREPROCESS*** function and then calls ***RECOMMENDATION*** function and returns **Recommendation_list**.

Now recommendation is done. After user action **(Click/Ignore)**, we need to update our Q table accordingly. This is where RL captures the **dynamic behaviour** of the user. If user clciks the product, then recommendng that product is good in that state so reward is given to that action in that state and Q value is updated. Similarly if ignored then penalty.
This is done by ***TABLE_UPDATE*** function in the code. It takes Click list as input and update the Q values.TABLE_UPDATE function calls tow more functions- ***UPDATE_CLICK_Q_VALUE*** and ***UPDATE_NOT_CLICK_Q_VALUE*** and update accordingly.

Now how will the new state be formed after user has clicked the recommended item?
As we have multiple sub state in our single state so, a sub state will go in a new state and all new sub state forms the next new State.

##### Example-
**State**=[article1, article2,article3]
If 3 product recommended was article9 which is clicked by user so our new state is
**new_State**=[article2,article3,article9]
now we call the ***PREPROCESS(new_State)*** function, it gives the sub states table as output as explained above. Now we check to which sub state that article9 belongs to, that is the new sub state of that sub state in which article9 was recommended. We give reward to that sub state and penalty to the remaining sub state whose recommended article was not clicked.
This is how update is done. Now after the new state is formed after the user click, according to new state we again provide the Recommendation list according to new state.

![image](https://cdn-media-1.freecodecamp.org/images/oQPHTmuB6tz7CVy3L05K1NlBmS6L8MUkgOud)

##### Things to note--
1.There is **no state of the user**, the states are of the agent,(Recommendation system).Every user come in any of the State we have defined, we need to know the state of the user he is in to recommend accordingly.
2.We also need to store the Recommended_List and Table given by ***PREPROCESS*** function of that user state. This is done to update the Q value after user action.
3.State is stored to recommend accordingly.
Now further explaination is about the code functions-
We need to recommend to a user, What information we need?
user's previous three history as State. This goes as input into ***RECOMMENDATION_LIST*** function which returns the **Recommendation_List.**
S be the State
##### Pseudo_Code
RECOMMENDATION LIST(S)
  > PREPROCESS(S)------------------------------------returns table as show above
   For 0-len(table) do:
        a=RECOMMENDATION(Table[i])----------------recommendation function where algorithm is implemented
        Rec_list.append(a)-----------------------------------stores the recommedation
   End for
> End do

***PREPROCESS(S)***-This takes input as State, output as table as explained above.
[article names, genre, State_Space, Q_table of that genre, row no of that state in it's table]
These informations are required in algorithm to decide recommendation.
 ***RECOMMENDATION*** function take input as **Table[i]** and returns Recommendation according to explore or exploit decided by algorithm.

As we know RL learns from feedback data.

##### UPDATE of Q_VALUES
Now as a user is recommended the Rec_List.We take implicit feedback to update our Q table
How do we do that?  
**Click-List** that stores the index of the items from **Recommendation_List** which were clicked by a particular user.    
To update the table we need the **Recommendation_List** and its previous state in which this **Recommendation_List** was provided. So for this reason we need to store the table and **Recommendation_List**
As we hae only click list, we need to make **Not_Click** list too, Because thos items were also recommended to user in that state only which was not clicked, it means recommending those items in that state is not fruitful so, we penalize those action in that state in their respective table.
##### Pseudo_Code
>function-UPDATE_TABLE(Click)
UPDATE_TABLE(Click): do
    Makes Not_Clicked list
    CLICK_Q_VALUE_UPDATE(reward, Click)
    Not_CLICK_Q_VALUE_UPDATE(-reward, Not_Click)
    RECOMMENDATION(New_State)
>End do

Now as the user has clicked some product his/her state will change that is done in ***CLICK_Q_VALUE_UPDATE***, ***NOT_CLICK_Q_VALUE_UPDATE*** function. After all updates is done, we get new state of the user  and the at last according to new state we give new recommendation list to that user.
For updating state-we use NEW_STATE function

### Dataset

Our problem statement was to make an article recommendation system using reinforcement learning. For reinforcement learning to work,we need user **feedback data**. Since we didn't have any users' feedback data for articles so instead of that I used 10M MovieLens **[dataset](https://grouplens.org/datasets/movielens/10m/)** as it had user feedback data too. I used movie with userId upto 150. The movies in it had multiple genre,I kept single genre of each movie as our dataset(Cleanipedia) had only one genre of each article. This data is used to optimize the table.

We made state from that data, then get the recommendation of that state from the RL recommendation and then check, which book was read by user in that state. If that book lies in the recommended list then it's good nad reward is assigned to it. If not then penalty was given

##### Optimization of table
In most of the cases it didn't happen because the data was random and there is very less chance that it matched with the recommendations. So if we assign negative reward to all the it will get to know these are not good to recommend but it won't know which is better to recommend, so what I did is that, I took the recommendation from RL then checked it with the actual read by the user in that state, if it's there then reward and penalty to others. If it's not in recommended list then we appended it into recommended list and gave reward to it and penalty to others. In this way it get to know which is good to recommend in this state along with which is not good to recommend.

### Installation

Python file requires [Anaconda](https://anaconda.org/) Prompt to run.
Go to the location where you have kept all the files and data together.
Install the dependencies and devDependencies.

```sh
 pip install --upgrade -r Q_Tablerequirements.txt
```

For executing code...

```sh
python Multiple_table_Argumentparser.py
```

#####  Working
When you execute this code, you will get some output
This code also optimizes the Q table using the data we generated. Now you can just call the ***RECOMMENDATION_LIST*** function for recommendations and the after user feedback call  ***TABLE_UPDATE*** function.
