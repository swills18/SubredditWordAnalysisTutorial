# Determining Buzz Words on Reddit



If there were a website that was basically 'the internet' in nutshell, it's Reddit. Reddit is a social media website with a distinct forum-like style. The website is broken into smaller, more specific sites called subreddits. Each subreddit has its own theme or topic for discussion. It can feel like navigating a jungle trying to find the right subreddit, but if you can imagine one it almost certainly exists.

'Hot' posts are at the front and center of a given subreddit. 'Hotness' is determined by both a post's score (upvotes minus downvotes) and how recent the post is. Reddit has this system to constantly cycle old posts out and new posts in so that the site does not stagnate for too long. But for every popular post, there are dozens of unpopular ones that most people never see. This begs the question: is there a pattern between posts' word choice and how much karma that post receives?

Information like this about a subreddit is important for ad companies to know how to target their ads. You shouldn't be posting an ad for Budweiser on r/stopdrinking or r/addiction. And knowing 'buzz words' can give an ad an edge; take for instance a gaming subreddit like r/league. Corporations may or may not be familiar with the latest gamer lingo so expanding their vocabulary is essential to get their message to an audience that tends to roll their eyes at ads.

To get started, we will need to be able to access Reddit using its API, PRAW.
More details can be found here: https://praw.readthedocs.io/en/latest/getting_started/quick_start.html


```python
!pip3 install praw
```

    Requirement already satisfied: praw in /opt/conda/lib/python3.8/site-packages (7.1.0)
    Requirement already satisfied: update-checker>=0.17 in /opt/conda/lib/python3.8/site-packages (from praw) (0.18.0)
    Requirement already satisfied: prawcore<2.0,>=1.3.0 in /opt/conda/lib/python3.8/site-packages (from praw) (1.5.0)
    Requirement already satisfied: websocket-client>=0.54.0 in /opt/conda/lib/python3.8/site-packages (from praw) (0.57.0)
    Requirement already satisfied: requests>=2.3.0 in /opt/conda/lib/python3.8/site-packages (from update-checker>=0.17->praw) (2.24.0)
    Requirement already satisfied: six in /opt/conda/lib/python3.8/site-packages (from websocket-client>=0.54.0->praw) (1.15.0)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/lib/python3.8/site-packages (from requests>=2.3.0->update-checker>=0.17->praw) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.8/site-packages (from requests>=2.3.0->update-checker>=0.17->praw) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.8/site-packages (from requests>=2.3.0->update-checker>=0.17->praw) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.8/site-packages (from requests>=2.3.0->update-checker>=0.17->praw) (1.25.10)



```python
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as BSoup
import re
import seaborn as sns
import sklearn.feature_extraction as skfeat
import matplotlib as plt
```


```python
import praw
```

Here is where we will need PRAW. Using this on your own requires that you first set up a Reddit account and then create an app on their developers' page. But for now, this tutorial will use an app I created on a burner account. The code below does not contain personal information as it does not submit any requests to the server that are specific to an account. An extension to this tutorial using account login info might involve creating a bot to automatically post/update information to Reddit. Perhaps with more machine learning, those posts could even be constructed from the set of popular words for a particular subreddit.


```python
#NOTE: not logged in with an account but this code accesses an app attached to a throwaway acc.
reddit = praw.Reddit(client_id="WIBYrJVVYt3WMQ", client_secret="8rbRMd75jk0VfuYpAVxkEIlxUSwm0A", user_agent="PostGetter")
```

First, let's try looking at some figures from r/all.
Again, PRAW API can be found here: https://praw.readthedocs.io/en/latest/getting_started/quick_start.html
Here, I need to get the data from Reddit into a DataFrame. There is a way to do this without creating the variable "hot_data", but it is more time-efficient to create a DataFrame from a large array of dicts than it is to create it once and append each hot post one-at-a-time. Posts are referred to as 'submissions' in the API. Good data points to collect are 'title', 'score', and 'timestamp' for this investigation.


```python
hot_posts = reddit.subreddit("all").hot(limit=200)

hot_data = []
i = 0
for post in hot_posts:
    #d_votes = int(post.score*(1-post.upvote_ratio)/(2*post.upvote_ratio-1) + 0.5)
    hot_data.append({"title":post.title,"score":post.score, #"upvotes":post.score+d_votes, "downvotes":d_votes,
                   "upvoteRatio":post.upvote_ratio,
                   "timestamp":pd.Timestamp(post.created_utc, unit="s"), 
                   "numComments":post.num_comments, "subreddit":post.subreddit})
    i += 1
hot_df = pd.DataFrame.from_dict(hot_data)
hot_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>score</th>
      <th>upvoteRatio</th>
      <th>timestamp</th>
      <th>numComments</th>
      <th>subreddit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kelly Loeffler says she uses private jet to sa...</td>
      <td>12884</td>
      <td>0.97</td>
      <td>2020-12-21 11:37:49</td>
      <td>428</td>
      <td>politics</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3 Dogs 1 Cat</td>
      <td>28738</td>
      <td>0.98</td>
      <td>2020-12-21 10:38:44</td>
      <td>273</td>
      <td>aww</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Russian opposition leader Alexey Navalny dupes...</td>
      <td>7067</td>
      <td>0.98</td>
      <td>2020-12-21 12:14:53</td>
      <td>267</td>
      <td>worldnews</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The concept of light pollution is crazy</td>
      <td>59310</td>
      <td>0.96</td>
      <td>2020-12-21 09:30:35</td>
      <td>1299</td>
      <td>interestingasfuck</td>
    </tr>
    <tr>
      <th>4</th>
      <td>That’s a big full brother</td>
      <td>16439</td>
      <td>0.94</td>
      <td>2020-12-21 10:48:55</td>
      <td>59</td>
      <td>PrequelMemes</td>
    </tr>
  </tbody>
</table>
</div>



If you did it right, it should look something like this. Note that `created_utc` is in seconds and that NSFW posts *might* get in the top 200 posts. I didn't care to here but you can filter them out by checking each post for `over_18`. If you need the total number of votes on a post, you can calculate upvotes and downvotes by the following:

`downvotes = score * (1 - ratio)/(2 * ratio - 1)`

`upvotes = score + downvotes`

`total votes = downvotes+upvotes`


```python
ax = sns.lineplot(data=hot_df, x="timestamp", y="score")
ax.set_title("Post score vs. Time post was created")
ax.tick_params(axis='x', labelrotation=45)
```


![png](output_11_0.png)


The plot above is a quick post score over time plot. Note that the x-axis is in UTC/UK Greenwich time. This can be avoided by modifying the timestamp constructor to add `tz = "[timezone]"`


```python
hot_df["titleLength"] = hot_df["title"].apply(lambda x:len(x))
sns.scatterplot(data=hot_df, x="titleLength", y="score").set_title("Number of characters in title vs. Post score")
```




    Text(0.5, 1.0, 'Number of characters in title vs. Post score')




![png](output_13_1.png)


This plot shows a correlation between title length and score. The data appears to take an 'L'-shape meaning that most 'hot' posts with long titles are generally not likely to have as high of a score as posts with short titles. It's a bit faint, but it also appears to have a negative corellation between title length and post score. 
This also shows that the majority of 'hot' posts have short titles (1-100 characters).

## r/Eyebleach

~~r/Aww is a subreddit dedicated to cute pictures. The popular joke about 'cat pictures on the internet' can probably be blamed on this subreddit. That being said, this sub will be a good example for how to (roughly) get the most popular terms on a given subreddit. The subreddit is primarily an image board so the titles will mostly be short, and there is a good chance many of the posts are similar to one another so they end up using the same words.~~

Post-project note: r/Aww is an absolutely terrible subreddit for reasons you can discover by looking at comments of some posts. r/Eyebleach is better for *ethically-sourced* cute pictures, so we will be using posts from that subreddit instead.

First, let's get the top posts in a very similar way as before, except now grabbing the comments for later. Avoid stickied posts since they aren't 'hot' and toss everything into a DataFrame.


```python
eb_posts = reddit.subreddit("Eyebleach").hot(limit=200)

eb_data = []
i = 0
for post in eb_posts:
    if(not post.stickied):
        eb_data.append({"title":post.title,"score":post.score,
                   "upvoteRatio":post.upvote_ratio,
                   "timestamp":pd.Timestamp(post.created_utc, unit="s"), 
                   "numComments":post.num_comments, "topComments":post.comments[0:5]})
    i += 1
eb_df = pd.DataFrame.from_dict(eb_data)
eb_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>score</th>
      <th>upvoteRatio</th>
      <th>timestamp</th>
      <th>numComments</th>
      <th>topComments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Just a little hamster enjoying some snacks in ...</td>
      <td>19256</td>
      <td>0.98</td>
      <td>2020-12-21 04:23:03</td>
      <td>67</td>
      <td>[ggk6rcs, ggka2rh, ggk6w8c, ggk7kul, ggk9r5n]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The perfect spot</td>
      <td>1186</td>
      <td>0.99</td>
      <td>2020-12-21 11:57:01</td>
      <td>14</td>
      <td>[ggkz37z, ggkwdej, ggkxiha, ggl2pnc, ggl10x7]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cutest wild kitten</td>
      <td>728</td>
      <td>0.99</td>
      <td>2020-12-21 10:55:02</td>
      <td>11</td>
      <td>[ggkrbdf, ggl6xjq, ggkr5do, ggkwran, ggl8cje]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The dog sheltered little kittens</td>
      <td>29976</td>
      <td>0.97</td>
      <td>2020-12-20 19:09:26</td>
      <td>94</td>
      <td>[ggipc24, ggj42fq, ggiqu2r, ggja9q4, ggjc3c7]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Meet Kiddo, any love for mixed dogs?</td>
      <td>481</td>
      <td>0.99</td>
      <td>2020-12-21 10:28:18</td>
      <td>26</td>
      <td>[ggkq1nn, ggkpmza, ggkqttg, ggkr28v, ggkugnu]</td>
    </tr>
  </tbody>
</table>
</div>




```python

ax = sns.violinplot(data=eb_df, x="upvoteRatio", palette="husl", alpha=0.5)
ax = sns.violinplot(data=hot_df, x="upvoteRatio", alpha=0.5)
ax.set_title("Upvote ratio of r/all vs. r/Eyebleach")
ax.legend(["r/all", "r/Eyebleach"])
```




    <matplotlib.legend.Legend at 0x7f5ab36e5700>




![png](output_18_1.png)


Above is a plot of the upvote ratios for both r/Eyebleach and r/all. Matplotlib is being disagreeable and it is portraying the legend with the colors of the lines in the middle rather than the colors of the violins themselves. Pink is r/Eyebleach and blue is r/all.

Next is r/Eyebleach vs. r/Aww, just to show the difference in upvote ratio


```python
aww_posts = reddit.subreddit("aww").hot(limit=200)

aww_data = []
i = 0
for post in aww_posts:
    if(not post.stickied):
        aww_data.append({"title":post.title,"score":post.score,
                   "upvoteRatio":post.upvote_ratio,
                   "timestamp":pd.Timestamp(post.created_utc, unit="s"), 
                   "numComments":post.num_comments, "topComments":post.comments[0:5]})
    i += 1
aww_df = pd.DataFrame.from_dict(aww_data)
aww_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>score</th>
      <th>upvoteRatio</th>
      <th>timestamp</th>
      <th>numComments</th>
      <th>topComments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hard to get all those wrinkles out of the sheets</td>
      <td>28810</td>
      <td>0.96</td>
      <td>2020-12-21 11:23:23</td>
      <td>416</td>
      <td>[ggkw9lh, ggkvio3, ggku47w, ggkvqx2, ggkvuhr]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3 Dogs 1 Cat</td>
      <td>65550</td>
      <td>0.96</td>
      <td>2020-12-21 10:38:44</td>
      <td>542</td>
      <td>[ggkq34t, ggktl5d, ggku306, ggkvb25, ggktp6k]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>They got a pup from the shelter. This was the ...</td>
      <td>5503</td>
      <td>0.98</td>
      <td>2020-12-21 13:17:07</td>
      <td>40</td>
      <td>[ggl7qsi, ggln90t, ggldflx, ggl6n8v, ggl7dza]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fluff</td>
      <td>16972</td>
      <td>0.98</td>
      <td>2020-12-21 09:57:32</td>
      <td>89</td>
      <td>[ggkq31r, ggkt7rq, ggkupe5, ggkntw6, ggkx1b5]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Doggo in its natural habitat</td>
      <td>10535</td>
      <td>0.99</td>
      <td>2020-12-21 09:39:49</td>
      <td>76</td>
      <td>[ggkok0l, ggkwwcc, ggks5qu, ggl5m5c, ggkosjm]</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = sns.violinplot(data=eb_df, x="upvoteRatio", palette="husl", alpha=0.5)
ax = sns.violinplot(data=aww_df, x="upvoteRatio", alpha=0.5)
ax.set_title("Upvote ratio of r/aww vs. r/Eyebleach")
ax.legend(["r/aww", "r/Eyebleach"])
```




    <matplotlib.legend.Legend at 0x7f5ab15fba60>




![png](output_21_1.png)


Again, pink is r/Eyebleach but blue ir r/aww. The two subreddits appear to have similar statistics as to how often users in each tend to upvote/downvote posts. However, r/aww has had an issue with bots farming karma on the subreddit recently, and looking at how narrow the upvote ratio is, it certainly could be possible. These are two subreddits with almost the exact same purpose, yet one has a wider distribution than the other. Not to mention how overwhelmingly negative the *top* responses are to some of the posts on r/Eyebleach... it's very fishy. To be safe, I am using r/Eyebleach moving forward.

In any case, next let's convert the comment IDs into comment text. You can refit this to use more than 5 comments per post, but it will take longer to run. I'm also getting the comment's score while I'm at it.


```python
eb_df = eb_df[["title", "score", "numComments", "topComments"]]
# eb_comments["topComments"].apply(lambda x: x)
for i in range(5):
    eb_df["Comment "+str(i+1)+" Text"] = eb_df["topComments"].apply(lambda x: np.nan if not len(x) > i else x[i].body)
    eb_df["Comment "+str(i+1)+" Score"] = eb_df["topComments"].apply(lambda x: np.nan if not len(x) > i else x[i].score)
eb_df = eb_df.drop("topComments", axis=1)
eb_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>score</th>
      <th>numComments</th>
      <th>Comment 1 Text</th>
      <th>Comment 1 Score</th>
      <th>Comment 2 Text</th>
      <th>Comment 2 Score</th>
      <th>Comment 3 Text</th>
      <th>Comment 3 Score</th>
      <th>Comment 4 Text</th>
      <th>Comment 4 Score</th>
      <th>Comment 5 Text</th>
      <th>Comment 5 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Just a little hamster enjoying some snacks in ...</td>
      <td>19256</td>
      <td>67</td>
      <td>Me_irl</td>
      <td>147.0</td>
      <td>What a cute little niblet!</td>
      <td>70.0</td>
      <td>monch</td>
      <td>41.0</td>
      <td>This has to be the most cute you can pack in b...</td>
      <td>53.0</td>
      <td>Can I please order a cup of hamster?</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The perfect spot</td>
      <td>1186</td>
      <td>14</td>
      <td>*If I fits, I sits.*</td>
      <td>33.0</td>
      <td>That’s adorable</td>
      <td>11.0</td>
      <td>Is that at an optometrist or something?  What'...</td>
      <td>12.0</td>
      <td>I found out that dogs and cats can get along j...</td>
      <td>5.0</td>
      <td>This is my dream</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cutest wild kitten</td>
      <td>728</td>
      <td>11</td>
      <td>Ocelot?</td>
      <td>5.0</td>
      <td>Babou?!</td>
      <td>3.0</td>
      <td>Good thing he's not any bigger!  He looks sooo...</td>
      <td>1.0</td>
      <td>Furfect</td>
      <td>1.0</td>
      <td>Rawr!</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The dog sheltered little kittens</td>
      <td>29976</td>
      <td>94</td>
      <td>Huskies can be very cat-like.</td>
      <td>560.0</td>
      <td>"These are my puppies. They are strange and me...</td>
      <td>428.0</td>
      <td>That's a big kitten</td>
      <td>184.0</td>
      <td>I want the one that looks like he had a piece ...</td>
      <td>201.0</td>
      <td>“Alright you all are going to be the meanest p...</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Meet Kiddo, any love for mixed dogs?</td>
      <td>481</td>
      <td>26</td>
      <td>Uh that's a fox not a dog. Super cute!</td>
      <td>40.0</td>
      <td>ALL the love for mixed dogs. Healthier, more v...</td>
      <td>23.0</td>
      <td>She looks like the most wonderful mix between ...</td>
      <td>7.0</td>
      <td>F o x g o b r r</td>
      <td>6.0</td>
      <td>That's an American dingo !  I have one too!</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Now for the fun part. Here is a Bag of Words/BOW using sklearn's count vectorizer (see more: https://scikit-learn.org/stable/modules/feature_extraction.html). The gist is that we need to fit the post titles to get a matrix of how many times a each word appears in each post. I'd recommend for the sake of convenience to use `stop_words='english'` to prevent words like 'the' from getting in.

For now, we will only be using post titles. Comments work the exact same way though, just with code to get all 5 columns. `eb_word_counts` is what we want: each word associated with how many times the word appears in all the post titles.


```python
vectorizer = skfeat.text.CountVectorizer(stop_words="english")

X = vectorizer.fit_transform(eb_df["title"])
# vectorizer.build_analyzer()
#vectorizer.get_feature_names()

eb_word_counts = {word: sum(X.toarray()[:,vectorizer.vocabulary_.get(word)]) for word in vectorizer.get_feature_names()}
eb_word_counts
```




    {'11': 1,
     '14': 1,
     '16yo': 1,
     '18': 1,
     '1950': 1,
     '1st': 1,
     'additions': 1,
     'adopted': 2,
     'aesthetic': 1,
     'aide': 1,
     'albus': 1,
     'apart': 1,
     'asleep': 1,
     'attack': 1,
     'australia': 1,
     'autumn': 1,
     'away': 2,
     'baby': 3,
     'ball': 2,
     'barn': 2,
     'barnabas': 1,
     'barren': 1,
     'bath': 2,
     'beach': 1,
     'beautiful': 2,
     'bed': 1,
     'beneath': 1,
     'bernardo': 1,
     'best': 4,
     'big': 1,
     'bike': 1,
     'bit': 3,
     'blanket': 2,
     'blankie': 1,
     'blem': 1,
     'blind': 1,
     'blue': 1,
     'boi': 2,
     'born': 1,
     'boy': 2,
     'brightens': 1,
     'bring': 1,
     'brings': 1,
     'broom': 1,
     'brother': 1,
     'brothers': 1,
     'buddies': 1,
     'buddy': 1,
     'bulldog': 1,
     'bun': 1,
     'bunny': 3,
     'burrito': 1,
     'cake': 1,
     'cakeday': 1,
     'card': 1,
     'care': 1,
     'caring': 1,
     'cat': 15,
     'catch': 1,
     'catnap': 1,
     'catnip': 2,
     'cats': 2,
     'caught': 2,
     'chameleon': 2,
     'change': 1,
     'character': 1,
     'check': 1,
     'chester': 1,
     'chewbug': 1,
     'china': 1,
     'christmas': 13,
     'chubby': 1,
     'circa': 1,
     'circles': 1,
     'class': 1,
     'claws': 1,
     'clay': 1,
     'climb': 1,
     'cold': 1,
     'collins': 1,
     'comfortable': 1,
     'comfy': 1,
     'coraline': 1,
     'couch': 2,
     'cozy': 1,
     'crazy': 1,
     'credit': 1,
     'cuddles': 1,
     'cuddling': 1,
     'cup': 1,
     'cut': 1,
     'cute': 7,
     'cuteness': 1,
     'cutest': 2,
     'cuz': 1,
     'dainty': 1,
     'dan': 1,
     'day': 3,
     'days': 3,
     'deaf': 2,
     'decided': 1,
     'decorate': 1,
     'decoration': 1,
     'decorations': 1,
     'defeated': 1,
     'derpy': 1,
     'dexter': 1,
     'dis': 1,
     'dog': 5,
     'doggo': 3,
     'doggy': 1,
     'doggytherapyy': 1,
     'dogloo': 1,
     'dogs': 2,
     'doing': 1,
     'dolphin': 1,
     'dolphins': 1,
     'don': 4,
     'doodle': 1,
     'door': 2,
     'duck': 1,
     'dumbledog': 1,
     'early': 1,
     'earned': 1,
     'eating': 1,
     'effective': 1,
     'emma': 1,
     'emu': 1,
     'enjoy': 1,
     'enjoying': 2,
     'evening': 1,
     'expression': 1,
     'eyes': 7,
     'face': 4,
     'faces': 1,
     'family': 2,
     'fell': 1,
     'felt': 2,
     'fern': 1,
     'ferret': 1,
     'finally': 2,
     'finch': 1,
     'finds': 1,
     'fine': 1,
     'finn': 1,
     'fit': 1,
     'fits': 1,
     'floofs': 1,
     'floofy': 1,
     'flowers': 1,
     'fluffy': 1,
     'food': 1,
     'fortress': 1,
     'foster': 1,
     'fren': 1,
     'friend': 6,
     'friends': 1,
     'fries': 1,
     'fuzzy': 1,
     'gen': 1,
     'gentle': 1,
     'gentleman': 1,
     'gets': 1,
     'getting': 1,
     'girl': 4,
     'giving': 2,
     'goes': 1,
     'going': 1,
     'goldfinch': 1,
     'good': 5,
     'got': 7,
     'grandparents': 1,
     'great': 2,
     'greeted': 1,
     'grew': 1,
     'grooming': 1,
     'guinea': 1,
     'guy': 1,
     'guys': 1,
     'halfway': 1,
     'hamster': 1,
     'hand': 1,
     'handsome': 2,
     'hanging': 1,
     'happening': 1,
     'happy': 4,
     'hard': 1,
     'hat': 1,
     'hated': 1,
     'having': 2,
     'heart': 1,
     'hello': 5,
     'helped': 1,
     'hep': 1,
     'hey': 1,
     'hi': 2,
     'himb': 1,
     'hims': 1,
     'hip': 1,
     'hit': 1,
     'holder': 1,
     'holiday': 2,
     'holidays': 3,
     'home': 4,
     'homie': 2,
     'hooman': 1,
     'hope': 1,
     'hoping': 1,
     'hour': 1,
     'huskies': 1,
     'idk': 1,
     'inseparable': 1,
     'interesting': 1,
     'jasper': 1,
     'java': 1,
     'jaxy': 1,
     'jedi': 1,
     'joined': 1,
     'joy': 1,
     'just': 5,
     'keeping': 1,
     'kiddo': 1,
     'kiss': 1,
     'kisses': 1,
     'kitten': 7,
     'kittens': 1,
     'kitty': 5,
     'know': 3,
     'koda': 1,
     'lamb': 1,
     'lap': 1,
     'leaf': 1,
     'left': 1,
     'legs': 1,
     'lemme': 1,
     'lettuce': 1,
     'lewis': 1,
     'life': 2,
     'like': 4,
     'likes': 1,
     'lil': 3,
     'lilly': 1,
     'lilo': 1,
     'lion': 1,
     'little': 11,
     'lived': 1,
     'long': 1,
     'look': 1,
     'looking': 1,
     'loop': 1,
     'lord': 1,
     'lot': 1,
     'louie': 1,
     'love': 7,
     'loved': 1,
     'lovely': 1,
     'loves': 8,
     'maddie': 1,
     'main': 1,
     'mainland': 1,
     'make': 2,
     'mama': 1,
     'man': 2,
     'managed': 1,
     'marci': 1,
     'means': 1,
     'meatball': 1,
     'meet': 4,
     'mind': 1,
     'missed': 1,
     'mixed': 1,
     'mom': 3,
     'moment': 1,
     'monday': 1,
     'monkey': 1,
     'monster': 1,
     'month': 1,
     'months': 1,
     'moon': 1,
     'morning': 3,
     'mr': 1,
     'multicolored': 1,
     'mustache': 1,
     'mutt': 1,
     'named': 2,
     'nap': 4,
     'nappy': 1,
     'naps': 1,
     'needed': 1,
     'nephew': 1,
     'new': 13,
     'newly': 1,
     'nice': 2,
     'night': 2,
     'nose': 1,
     'notice': 1,
     'oakley': 1,
     'ok': 1,
     'old': 7,
     'ole': 1,
     'ollie': 1,
     'open': 1,
     'outside': 2,
     'pajamas': 1,
     'pancake': 1,
     'passing': 1,
     'paws': 1,
     'peekaboo': 1,
     'perfect': 1,
     'pet': 3,
     'phil': 1,
     'phone': 1,
     'photo': 2,
     'photos': 1,
     'picture': 3,
     'pictures': 1,
     'pig': 1,
     'piggies': 1,
     'pillow': 2,
     'pink': 2,
     'place': 1,
     'play': 2,
     'playful': 1,
     'playing': 1,
     'pokemon': 1,
     'polite': 1,
     'pops': 1,
     'possum': 1,
     'posted': 1,
     'precious': 1,
     'present': 2,
     'pretty': 1,
     'probably': 1,
     'problems': 1,
     'proper': 1,
     'protest': 1,
     'pup': 4,
     'pupper': 2,
     'puppy': 10,
     'pure': 1,
     'purrfect': 1,
     'quarantine': 1,
     'quick': 1,
     'quinn': 1,
     'quokka': 2,
     'ready': 4,
     'real': 1,
     'realizing': 1,
     'receive': 1,
     'reindeer': 1,
     'relax': 1,
     'relaxation': 1,
     'relaxing': 1,
     'rescue': 1,
     'resist': 1,
     'ride': 1,
     'riding': 1,
     'ripley': 1,
     'robe': 1,
     'rose': 1,
     'running': 2,
     'russians': 1,
     'safe': 1,
     'sanctuary': 2,
     'santa': 3,
     'santy': 1,
     'say': 3,
     'says': 1,
     'sberla996': 1,
     'scarf': 2,
     'schedule': 1,
     'school': 1,
     'scruffy': 1,
     'seal': 2,
     'season': 2,
     'selfie': 3,
     'separate': 1,
     'share': 1,
     'sharing': 1,
     'shelter': 2,
     'sheltered': 1,
     'shy': 1,
     'siberia': 1,
     'siblings': 1,
     'sierra': 1,
     'silly': 2,
     'sit': 1,
     'sits': 1,
     'sky': 1,
     'sleep': 1,
     'sleeping': 1,
     'sleeps': 1,
     'sleepy': 2,
     'sleeve': 1,
     'slow': 1,
     'small': 1,
     'smile': 3,
     'smiley': 1,
     'snacks': 1,
     'snaggle': 1,
     'snaggletooth': 1,
     'snakes': 1,
     'snooze': 1,
     'snuggling': 1,
     'son': 1,
     'soo': 1,
     'soul': 1,
     'source': 1,
     'spending': 1,
     'spirit': 1,
     'spot': 1,
     'stayed': 1,
     'sticks': 1,
     'stitch': 1,
     'sub': 1,
     'successful': 1,
     'sunday': 1,
     'sure': 1,
     'surprised': 1,
     'suspect': 1,
     'sweet': 3,
     'tail': 1,
     'taken': 1,
     'taking': 1,
     'tell': 1,
     'thing': 1,
     'think': 1,
     'time': 5,
     'tiny': 1,
     'today': 2,
     'told': 1,
     'toof': 1,
     'took': 2,
     'toooo': 1,
     'toothless': 1,
     'tough': 1,
     'town': 1,
     'toy': 1,
     'training': 1,
     'treats': 2,
     'tree': 4,
     'trick': 1,
     'tried': 2,
     'trought': 1,
     'trusts': 1,
     'try': 1,
     'trying': 1,
     'tucker': 1,
     'turned': 1,
     'typical': 1,
     'uncle': 1,
     'unfortunately': 1,
     'use': 1,
     'uses': 1,
     'using': 1,
     'vampire': 1,
     've': 1,
     'vibes': 1,
     'video': 1,
     'view': 1,
     'visual': 1,
     'waits': 1,
     'wake': 1,
     'wanna': 1,
     'wanted': 1,
     'wants': 1,
     'war': 1,
     'warm': 1,
     'warn': 1,
     'watson': 1,
     'week': 2,
     'wild': 1,
     'wink': 2,
     'woke': 1,
     'wonder': 1,
     'woof': 1,
     'world': 1,
     'xmass': 1,
     'year': 3,
     'years': 2,
     'zooby': 1,
     'zoom': 1}



Next we'll do a quick filter to get rid of the irrelevant terms.


```python
eb_word_counts = {k: v for k,v in eb_word_counts.items() if re.compile("[a-zA-Z]").match(k)}
#Docs list 're and 've like in you're and we've as not removed
if "re" in eb_word_counts:
    eb_word_counts.pop("re")
if "ve" in eb_word_counts:
    eb_word_counts.pop("ve")
eb_word_counts
```




    {'additions': 1,
     'adopted': 2,
     'aesthetic': 1,
     'aide': 1,
     'albus': 1,
     'apart': 1,
     'asleep': 1,
     'attack': 1,
     'australia': 1,
     'autumn': 1,
     'away': 2,
     'baby': 3,
     'ball': 2,
     'barn': 2,
     'barnabas': 1,
     'barren': 1,
     'bath': 2,
     'beach': 1,
     'beautiful': 2,
     'bed': 1,
     'beneath': 1,
     'bernardo': 1,
     'best': 4,
     'big': 1,
     'bike': 1,
     'bit': 3,
     'blanket': 2,
     'blankie': 1,
     'blem': 1,
     'blind': 1,
     'blue': 1,
     'boi': 2,
     'born': 1,
     'boy': 2,
     'brightens': 1,
     'bring': 1,
     'brings': 1,
     'broom': 1,
     'brother': 1,
     'brothers': 1,
     'buddies': 1,
     'buddy': 1,
     'bulldog': 1,
     'bun': 1,
     'bunny': 3,
     'burrito': 1,
     'cake': 1,
     'cakeday': 1,
     'card': 1,
     'care': 1,
     'caring': 1,
     'cat': 15,
     'catch': 1,
     'catnap': 1,
     'catnip': 2,
     'cats': 2,
     'caught': 2,
     'chameleon': 2,
     'change': 1,
     'character': 1,
     'check': 1,
     'chester': 1,
     'chewbug': 1,
     'china': 1,
     'christmas': 13,
     'chubby': 1,
     'circa': 1,
     'circles': 1,
     'class': 1,
     'claws': 1,
     'clay': 1,
     'climb': 1,
     'cold': 1,
     'collins': 1,
     'comfortable': 1,
     'comfy': 1,
     'coraline': 1,
     'couch': 2,
     'cozy': 1,
     'crazy': 1,
     'credit': 1,
     'cuddles': 1,
     'cuddling': 1,
     'cup': 1,
     'cut': 1,
     'cute': 7,
     'cuteness': 1,
     'cutest': 2,
     'cuz': 1,
     'dainty': 1,
     'dan': 1,
     'day': 3,
     'days': 3,
     'deaf': 2,
     'decided': 1,
     'decorate': 1,
     'decoration': 1,
     'decorations': 1,
     'defeated': 1,
     'derpy': 1,
     'dexter': 1,
     'dis': 1,
     'dog': 5,
     'doggo': 3,
     'doggy': 1,
     'doggytherapyy': 1,
     'dogloo': 1,
     'dogs': 2,
     'doing': 1,
     'dolphin': 1,
     'dolphins': 1,
     'don': 4,
     'doodle': 1,
     'door': 2,
     'duck': 1,
     'dumbledog': 1,
     'early': 1,
     'earned': 1,
     'eating': 1,
     'effective': 1,
     'emma': 1,
     'emu': 1,
     'enjoy': 1,
     'enjoying': 2,
     'evening': 1,
     'expression': 1,
     'eyes': 7,
     'face': 4,
     'faces': 1,
     'family': 2,
     'fell': 1,
     'felt': 2,
     'fern': 1,
     'ferret': 1,
     'finally': 2,
     'finch': 1,
     'finds': 1,
     'fine': 1,
     'finn': 1,
     'fit': 1,
     'fits': 1,
     'floofs': 1,
     'floofy': 1,
     'flowers': 1,
     'fluffy': 1,
     'food': 1,
     'fortress': 1,
     'foster': 1,
     'fren': 1,
     'friend': 6,
     'friends': 1,
     'fries': 1,
     'fuzzy': 1,
     'gen': 1,
     'gentle': 1,
     'gentleman': 1,
     'gets': 1,
     'getting': 1,
     'girl': 4,
     'giving': 2,
     'goes': 1,
     'going': 1,
     'goldfinch': 1,
     'good': 5,
     'got': 7,
     'grandparents': 1,
     'great': 2,
     'greeted': 1,
     'grew': 1,
     'grooming': 1,
     'guinea': 1,
     'guy': 1,
     'guys': 1,
     'halfway': 1,
     'hamster': 1,
     'hand': 1,
     'handsome': 2,
     'hanging': 1,
     'happening': 1,
     'happy': 4,
     'hard': 1,
     'hat': 1,
     'hated': 1,
     'having': 2,
     'heart': 1,
     'hello': 5,
     'helped': 1,
     'hep': 1,
     'hey': 1,
     'hi': 2,
     'himb': 1,
     'hims': 1,
     'hip': 1,
     'hit': 1,
     'holder': 1,
     'holiday': 2,
     'holidays': 3,
     'home': 4,
     'homie': 2,
     'hooman': 1,
     'hope': 1,
     'hoping': 1,
     'hour': 1,
     'huskies': 1,
     'idk': 1,
     'inseparable': 1,
     'interesting': 1,
     'jasper': 1,
     'java': 1,
     'jaxy': 1,
     'jedi': 1,
     'joined': 1,
     'joy': 1,
     'just': 5,
     'keeping': 1,
     'kiddo': 1,
     'kiss': 1,
     'kisses': 1,
     'kitten': 7,
     'kittens': 1,
     'kitty': 5,
     'know': 3,
     'koda': 1,
     'lamb': 1,
     'lap': 1,
     'leaf': 1,
     'left': 1,
     'legs': 1,
     'lemme': 1,
     'lettuce': 1,
     'lewis': 1,
     'life': 2,
     'like': 4,
     'likes': 1,
     'lil': 3,
     'lilly': 1,
     'lilo': 1,
     'lion': 1,
     'little': 11,
     'lived': 1,
     'long': 1,
     'look': 1,
     'looking': 1,
     'loop': 1,
     'lord': 1,
     'lot': 1,
     'louie': 1,
     'love': 7,
     'loved': 1,
     'lovely': 1,
     'loves': 8,
     'maddie': 1,
     'main': 1,
     'mainland': 1,
     'make': 2,
     'mama': 1,
     'man': 2,
     'managed': 1,
     'marci': 1,
     'means': 1,
     'meatball': 1,
     'meet': 4,
     'mind': 1,
     'missed': 1,
     'mixed': 1,
     'mom': 3,
     'moment': 1,
     'monday': 1,
     'monkey': 1,
     'monster': 1,
     'month': 1,
     'months': 1,
     'moon': 1,
     'morning': 3,
     'mr': 1,
     'multicolored': 1,
     'mustache': 1,
     'mutt': 1,
     'named': 2,
     'nap': 4,
     'nappy': 1,
     'naps': 1,
     'needed': 1,
     'nephew': 1,
     'new': 13,
     'newly': 1,
     'nice': 2,
     'night': 2,
     'nose': 1,
     'notice': 1,
     'oakley': 1,
     'ok': 1,
     'old': 7,
     'ole': 1,
     'ollie': 1,
     'open': 1,
     'outside': 2,
     'pajamas': 1,
     'pancake': 1,
     'passing': 1,
     'paws': 1,
     'peekaboo': 1,
     'perfect': 1,
     'pet': 3,
     'phil': 1,
     'phone': 1,
     'photo': 2,
     'photos': 1,
     'picture': 3,
     'pictures': 1,
     'pig': 1,
     'piggies': 1,
     'pillow': 2,
     'pink': 2,
     'place': 1,
     'play': 2,
     'playful': 1,
     'playing': 1,
     'pokemon': 1,
     'polite': 1,
     'pops': 1,
     'possum': 1,
     'posted': 1,
     'precious': 1,
     'present': 2,
     'pretty': 1,
     'probably': 1,
     'problems': 1,
     'proper': 1,
     'protest': 1,
     'pup': 4,
     'pupper': 2,
     'puppy': 10,
     'pure': 1,
     'purrfect': 1,
     'quarantine': 1,
     'quick': 1,
     'quinn': 1,
     'quokka': 2,
     'ready': 4,
     'real': 1,
     'realizing': 1,
     'receive': 1,
     'reindeer': 1,
     'relax': 1,
     'relaxation': 1,
     'relaxing': 1,
     'rescue': 1,
     'resist': 1,
     'ride': 1,
     'riding': 1,
     'ripley': 1,
     'robe': 1,
     'rose': 1,
     'running': 2,
     'russians': 1,
     'safe': 1,
     'sanctuary': 2,
     'santa': 3,
     'santy': 1,
     'say': 3,
     'says': 1,
     'sberla996': 1,
     'scarf': 2,
     'schedule': 1,
     'school': 1,
     'scruffy': 1,
     'seal': 2,
     'season': 2,
     'selfie': 3,
     'separate': 1,
     'share': 1,
     'sharing': 1,
     'shelter': 2,
     'sheltered': 1,
     'shy': 1,
     'siberia': 1,
     'siblings': 1,
     'sierra': 1,
     'silly': 2,
     'sit': 1,
     'sits': 1,
     'sky': 1,
     'sleep': 1,
     'sleeping': 1,
     'sleeps': 1,
     'sleepy': 2,
     'sleeve': 1,
     'slow': 1,
     'small': 1,
     'smile': 3,
     'smiley': 1,
     'snacks': 1,
     'snaggle': 1,
     'snaggletooth': 1,
     'snakes': 1,
     'snooze': 1,
     'snuggling': 1,
     'son': 1,
     'soo': 1,
     'soul': 1,
     'source': 1,
     'spending': 1,
     'spirit': 1,
     'spot': 1,
     'stayed': 1,
     'sticks': 1,
     'stitch': 1,
     'sub': 1,
     'successful': 1,
     'sunday': 1,
     'sure': 1,
     'surprised': 1,
     'suspect': 1,
     'sweet': 3,
     'tail': 1,
     'taken': 1,
     'taking': 1,
     'tell': 1,
     'thing': 1,
     'think': 1,
     'time': 5,
     'tiny': 1,
     'today': 2,
     'told': 1,
     'toof': 1,
     'took': 2,
     'toooo': 1,
     'toothless': 1,
     'tough': 1,
     'town': 1,
     'toy': 1,
     'training': 1,
     'treats': 2,
     'tree': 4,
     'trick': 1,
     'tried': 2,
     'trought': 1,
     'trusts': 1,
     'try': 1,
     'trying': 1,
     'tucker': 1,
     'turned': 1,
     'typical': 1,
     'uncle': 1,
     'unfortunately': 1,
     'use': 1,
     'uses': 1,
     'using': 1,
     'vampire': 1,
     'vibes': 1,
     'video': 1,
     'view': 1,
     'visual': 1,
     'waits': 1,
     'wake': 1,
     'wanna': 1,
     'wanted': 1,
     'wants': 1,
     'war': 1,
     'warm': 1,
     'warn': 1,
     'watson': 1,
     'week': 2,
     'wild': 1,
     'wink': 2,
     'woke': 1,
     'wonder': 1,
     'woof': 1,
     'world': 1,
     'xmass': 1,
     'year': 3,
     'years': 2,
     'zooby': 1,
     'zoom': 1}



Now, let's plot the most popular terms in r/Eyebleach.


```python
high_temp_counts = {k:v for k, v in eb_word_counts.items() if v > 4}
# ax = plt.pyplot.barh(y=[k for k in range(len(max_temp_counts))],
#                     width=max_temp_counts.values(), tick_label=[k for k in max_temp_counts.keys()])

# sns.barplot(x=eb_word_counts.keys(), y=eb_word_counts.values)
sns.barplot(y=[k for k in high_temp_counts.keys()], x=[v for v in high_temp_counts.values()], orient="h").set_title("Most popular Words on r/Eyebleach (5+ uses)")
```




    Text(0.5, 1.0, 'Most popular Words on r/Eyebleach (5+ uses)')




![png](output_29_1.png)


And again, for good measure:


```python
max_temp_counts = {k:v for k, v in eb_word_counts.items() if v > 7}
# ax = plt.pyplot.barh(y=[k for k in range(len(max_temp_counts))],
#                     width=max_temp_counts.values(), tick_label=[k for k in max_temp_counts.keys()])

# sns.barplot(x=eb_word_counts.keys(), y=eb_word_counts.values)
sns.barplot(y=[k for k in max_temp_counts.keys()], x=[v for v in max_temp_counts.values()], orient="h").set_title("Most popular terms on r/Eyebleach (8+ uses)")
```




    Text(0.5, 1.0, 'Most popular terms on r/Eyebleach (8+ uses)')




![png](output_31_1.png)


At the current date and time, it appears 'cat', 'christmas', 'little', 'puppy', and 'new' are all pretty common words.
This is to be expected, after all that's what the subreddit is about. Unfortunately, there is no way to tell what is in the images themselves without an advanced algorithm trained to differentiate between cats and dogs, but this is a good place to start. 

On this particular subreddit, the most popular words do not mean a whole lot since they are fairly predictable. Next, let's try and see how 'sophisticated' the vocabulary of r/Eyebleach users is. Note that this takes from all the words, not the most popular ones.


```python
temp_len_counts = {len(k):0 for k in eb_word_counts.keys()}
for k, v in eb_word_counts.items():
    temp_len_counts[len(k)] += v
temp_len_counts = {k:temp_len_counts[k] for k in sorted(temp_len_counts)}
temp_len_counts

# plt.pyplot.hist(x=[k for k in temp_len_counts.keys()], bins=1, orientation="horizontal")
# ax = plt.pyplot.barh(y=[k for k in range(len(temp_len_counts))],
#                     width=temp_len_counts.values(), tick_label=[k for k in temp_len_counts.keys()])
ax = sns.barplot(x=[k for k in temp_len_counts.keys()], y=[v for v in temp_len_counts.values()])
ax.set_title("Word length in titles of r/EyeBleach post titles")
ax.set_xlabel("Word length")
ax.set_ylabel("Occurences")
```




    Text(0, 0.5, 'Occurences')




![png](output_33_1.png)


Generally, it appears that words have ~4 letters in them.
We don't have any frame of reference, so let's use r/Science to show a clear disparity. The following few cells repeat all the same steps as before but with a new subreddit.


```python
sci_posts = reddit.subreddit("science").hot(limit=200)

sci_data = [] #[None for i in range(50)]
i = 0
for post in sci_posts:
    if(not post.stickied):
        sci_data.append({"title":post.title,"score":post.score,
                   "upvoteRatio":post.upvote_ratio,
                   "timestamp":pd.Timestamp(post.created_utc, unit="s"), 
                   "numComments":post.num_comments, "topComments":post.comments[0:5]})
    i += 1
sci_df = pd.DataFrame.from_dict(sci_data)
sci_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>score</th>
      <th>upvoteRatio</th>
      <th>timestamp</th>
      <th>numComments</th>
      <th>topComments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Quarter of world may not have access to COVID-...</td>
      <td>14932</td>
      <td>0.94</td>
      <td>2020-12-21 09:17:36</td>
      <td>784</td>
      <td>[ggkle1a, ggky0v7, ggkyfxx, ggkyxcx, ggkvw3q]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The COVID-19 pandemic masks the way people per...</td>
      <td>1721</td>
      <td>0.92</td>
      <td>2020-12-21 10:36:25</td>
      <td>255</td>
      <td>[ggkpvh6, ggkq8sw, ggkupfp, ggks23i, ggkxal4]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Republican lawmakers vote far more often again...</td>
      <td>313</td>
      <td>0.88</td>
      <td>2020-12-21 15:49:20</td>
      <td>36</td>
      <td>[gglf0rw, ggliv1v, ggljumt, gglka0y, gglm567]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>New study links psychopathic tendencies to rac...</td>
      <td>51631</td>
      <td>0.62</td>
      <td>2020-12-20 20:57:41</td>
      <td>3967</td>
      <td>[ggimj6v, ggixp8z, ggj1p1r, ggk9xru, ggiznsw]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alligators can regrow severed tails, surprisin...</td>
      <td>161</td>
      <td>0.95</td>
      <td>2020-12-21 13:05:19</td>
      <td>25</td>
      <td>[ggkzeqk, ggl4jlt, ggl49xv, ggl7fzd, gglg8ee]</td>
    </tr>
  </tbody>
</table>
</div>




```python
vec2 = skfeat.text.CountVectorizer(stop_words="english")

X2 = vec2.fit_transform(sci_df["title"])
# vectorizer.build_analyzer()
#vectorizer.get_feature_names()

sci_wc = {word: sum(X2.toarray()[:,vec2.vocabulary_.get(word)]) for word in vec2.get_feature_names()}


sci_wc = {k: v for k,v in sci_wc.items() if v > 1 and re.compile("[a-zA-Z]").match(k)}
#Docs list 're and 've like in you're and we've as not removed
if "re" in sci_wc:
    sci_wc.pop("re")
if "ve" in sci_wc:
    sci_wc.pop("ve")

sci_lens = {len(k):0 for k in sci_wc.keys()}
for k, v in sci_wc.items():
    sci_lens[len(k)] += v
sci_lens = {k:sci_lens[k] for k in sorted(sci_lens)}
```


```python
f, ax = plt.pyplot.subplots()

plt.pyplot.bar(x=[k for k in sci_lens.keys()], height=[v for v in sci_lens.values()], color="skyblue")
plt.pyplot.bar(x=[k for k in temp_len_counts.keys()], height=[v for v in temp_len_counts.values()], width=0.4, color="coral")

ax.set_title("Complexity of Submission titles on r/Aww and r/Science")
plt.pyplot.legend(["r/Science", "r/Eyebleach"])
```




    <matplotlib.legend.Legend at 0x7f5aac4ad490>




![png](output_37_1.png)


So here's what the data shows: r/Eyebleach has fewer words as indicated by the shorter bars, and r/Science has more words and is centered around a word length of ~6.


```python
aww_df = aww_df[["title", "score", "numComments", "topComments"]]
# aww_comments["topComments"].apply(lambda x: x)
for i in range(5):
    aww_df["Comment "+str(i+1)+" Text"] = aww_df["topComments"].apply(lambda x: np.nan if not len(x) > i else x[i].body)
    aww_df["Comment "+str(i+1)+" Score"] = aww_df["topComments"].apply(lambda x: np.nan if not len(x) > i else x[i].score)
aww_df = aww_df.drop("topComments", axis=1)
```


```python
aww_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>score</th>
      <th>numComments</th>
      <th>Comment 1 Text</th>
      <th>Comment 1 Score</th>
      <th>Comment 2 Text</th>
      <th>Comment 2 Score</th>
      <th>Comment 3 Text</th>
      <th>Comment 3 Score</th>
      <th>Comment 4 Text</th>
      <th>Comment 4 Score</th>
      <th>Comment 5 Text</th>
      <th>Comment 5 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hard to get all those wrinkles out of the sheets</td>
      <td>28810</td>
      <td>416</td>
      <td>So many cats yet no worrys about cat hair.</td>
      <td>1963.0</td>
      <td>I can’t make the bed with ONE cat in the room,...</td>
      <td>466.0</td>
      <td>What playful little skinned demons you have!</td>
      <td>287.0</td>
      <td>You leave those wrinkles alone! They're just t...</td>
      <td>211.0</td>
      <td>I love the lil simultaneous pounce when she pu...</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3 Dogs 1 Cat</td>
      <td>65550</td>
      <td>542</td>
      <td>Look at how gently they insert themselves into...</td>
      <td>3574.0</td>
      <td>This is cut too soon, I need to see the doggo ...</td>
      <td>1711.0</td>
      <td>I love this. When I was growing up my parents ...</td>
      <td>493.0</td>
      <td>We had two golden retrievers, and an old, cant...</td>
      <td>140.0</td>
      <td>That bed looks warm</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>They got a pup from the shelter. This was the ...</td>
      <td>5503</td>
      <td>40</td>
      <td>Shelter dogs (and cats) just appreciate things...</td>
      <td>117.0</td>
      <td>My roommates and I have been fostering doggos ...</td>
      <td>17.0</td>
      <td>No one can convince me that animals don't have...</td>
      <td>22.0</td>
      <td>Damn that a good boi</td>
      <td>8.0</td>
      <td>I hope the pup will live a happy and fulfilled...</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fluff</td>
      <td>16972</td>
      <td>89</td>
      <td>What eyeliner does she use?</td>
      <td>349.0</td>
      <td>This cat is still prettier than me even during...</td>
      <td>88.0</td>
      <td>What is that purple band all about?</td>
      <td>103.0</td>
      <td>Not fat, just fluffy</td>
      <td>76.0</td>
      <td>I never saw a cat with eyes that big</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Doggo in its natural habitat</td>
      <td>10535</td>
      <td>76</td>
      <td>Liondog</td>
      <td>112.0</td>
      <td>This looks like a scene from a movie where the...</td>
      <td>59.0</td>
      <td>Oh no, put a wool sock on his left paw, it’s c...</td>
      <td>67.0</td>
      <td>I have a snow dog as well. Different breed, an...</td>
      <td>17.0</td>
      <td>Greetings human, I am ice dog</td>
      <td>33.0</td>
    </tr>
  </tbody>
</table>
</div>



And here's another quick demonstration, this time just for top comments:


```python
vec3 = skfeat.text.CountVectorizer(stop_words="english")

X3 = vec3.fit_transform(eb_df[eb_df["Comment 1 Text"].notna()]["Comment 1 Text"])
# vectorizer.build_analyzer()
#vectorizer.get_feature_names()

eb_wc = {word: sum(X3.toarray()[:,vec3.vocabulary_.get(word)]) for word in vec3.get_feature_names()}


eb_wc = {k: v for k,v in eb_wc.items() if v > 1 and re.compile("[a-zA-Z]").match(k)}
#Docs list 're and 've like in you're and we've as not removed
if "re" in eb_wc:
    eb_wc.pop("re")
if "ve" in eb_wc:
    eb_wc.pop("ve")

eb_lens = {len(k):0 for k in eb_wc.keys()}
for k, v in eb_wc.items():
    eb_lens[len(k)] += v
eb_lens = {k:eb_lens[k] for k in sorted(eb_lens)}

high_temp_counts2 = {k:v for k, v in eb_wc.items() if v > 4}
# ax = plt.pyplot.barh(y=[k for k in range(len(max_temp_counts))],
#                     width=max_temp_counts.values(), tick_label=[k for k in max_temp_counts.keys()])

# sns.barplot(x=eb_word_counts.keys(), y=eb_word_counts.values)
sns.barplot(y=[k for k in high_temp_counts2.keys()], x=[v for v in high_temp_counts2.values()], orient="h").set_title("Most popular Words on r/Eyebleach Comments (5+ uses)")
```




    Text(0.5, 1.0, 'Most popular Words on r/Eyebleach Comments (5+ uses)')




![png](output_42_1.png)


About what you'd expect for a crowd of people looking at something cute.

For the next one, let's see how using the word 'cute' in the title correlates with the score of the post. Most terms do not show much correlation, but in this case it works.


```python
sns.violinplot(x=eb_df["score"], color="grey", bw=0.2)
ax = sns.violinplot(x=eb_df[eb_df["title"].apply(lambda x: "cute" in x)]["score"], bw=0.2)
ax.set_xlim([-1000,3000])
ax.set_title("Post score vs. usage of \'cute\' in a post\'s title")
```




    Text(0.5, 1.0, "Post score vs. usage of 'cute' in a post's title")




![png](output_44_1.png)


There is a notable group of outliers to the right. Those ones must be truly 'cute'.

## Conclusion

It seems like most of the results are to be expected. You can use this technique on any subreddit to get a picture of what the conversation is usually about, and what words may increase your chances of your posts being noticed. 

This is the kind of data advertisers might be interested in using, especially on reddit where ads blend in with other content very well. Data can also be obtained in a similar way to try to discover new words (If you look at the r/Eyebleach comments bar chart, many replies use colloquial words such as 'op' and 'pupper') or even to train a model to learn a dialogue-like English. 

Another important application of this technology is security. Certain normal words may be 'buzz words' in underground subreddits featuring illegal content. There won't always be eyes in dark corners, so using a bot to pick up on a popular word (i.e., 'cheese pizza/cp' is a well-known euphemism for 'child porn') can alert Reddit or authorities about illegal activity. Consequently, this also means that some users can use this technology to bombard subreddits with computer-generated posts to farm karma based on 'buzz words' alone. For instance, just auto-replying the Comment "Cute!" to r/Eyebleach might get some users free karma -- but it's also worth noting there are far more efficient ways to farm karma and these cons do not outweigh the pros. Regardless, word usage is important data for analytics and should be used responsibly. For the average-case user curious about data science and Reddit trends, this tutorial should be a good place to start.
