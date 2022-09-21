# OLA-Project
This project contains a simulation of Ferrero's online shop. The goal is to find the best price of each product in order to maximize the total final reward.

After having set an hypotethical cost of production c for each product, we decided to take as base price p = 5*c. For each product, we explored the four possible prices obtained respectively as (0.4p, 0.8p, 1.2p, 1.6p).

We divided the users in three classes: children, men and women. This division allowed us to provide each kind of user with a different demand curve. To determine the class, we have two binary features (children/adult, male/female), male and female children are placed in the class children while adult males and adult females are respectively men and women.

->![image](https://user-images.githubusercontent.com/79787310/191463513-aed681d6-21a7-4cef-9196-0f554715ca9f.png)<-

Every day, there is a random number of potential new customers. In particular, every single customer can land on the webpage in which one of the 5 products is primary or on the webpage of a product sold by a (non-strategic) competitor. To model this behavior, each day we sample from a Dirichlet distribution the proportions of users landing on each webpage . The parameters of the Dirichlet are different in each class, representing the case in which users of different class are more interested in different products. 

To simulate the decision of a customer when buying a product, we modeled her reservation price with a Gaussian random variable: when a customer is shown a product, her reservation price is sampled and if it is lower than the actual price the customer buys the product. 
