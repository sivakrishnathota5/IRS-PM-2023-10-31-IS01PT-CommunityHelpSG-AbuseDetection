# [ Practice Module ] Project Submission 

IRS-PM-2023-10-31-IS01PT-CommunityHelpSG.zip



### <<<<<<<<<<<<<<<<<<<< Start of Template >>>>>>>>>>>>>>>>>>>>

---

## SECTION 1 : PROJECT TITLE
## Singapore Community Help , 
### Social Network

<img src="https://github.com/sivakrishnathota5/IRS-PM-2023-10-31-IS01PT-CommunityHelpSG-AbuseDetection/blob/main/Miscellaneous/HELPWEBPAGE.png"
     style="float: left; margin-right: 0px;" />

---

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT
### Abuse Language Detection
Offensive Text is pervasive in social media. Individuals frequently take advantage of the perceived anonymity of computer-mediated communication, using this to engage in
behavior that many of them would not consider in real life. Online communities, social media platforms, and technology companies have been investing heavily in ways to cope with offensive language in the form of text to prevent abusive behavior
in social media. The proliferation of, social media platforms resulted in a
remarkable increase in user-generated content. These platforms have empowered users to create, share and exchange content for interacting and communicating with each other. However, these have also opened new avenues to cyberbullies
and haters who can spread their negativity to a larger audience, often anonymously. Due to the pervasiveness and severity of this behavior, many automated approaches that
employ natural language processing (NLP), machine learning and deep learning techniques have been proposed in the past. This survey offers an extensive overview of the stateof- the-art approaches proposed by the research community to identify offensive content. Based on our comprehensive literature survey, a categorization of different approaches and features employed by the researchers in the detection process
are presented. This survey also incorporates the major challenges that require considerable research efforts in this domain. Finally, future research directions with an aim of developing robust abusive content detection system for social
media are also discussed. A specific popular form of online harassment is the use
of abusive language. One abusive or toxic statement is being sent every 30 seconds across the globe. The use of abusive language on social media contributes to mental or emotional stress, with one in ten people developing such issues.These
abusive Tweets and comments detection and deletion in social media is more important.
“The Online Criminal Harms Act will empower the Singapore government to issue directions to individuals, entities, online service providers and app stores requiring them to remove or block access to potentially criminal content.

### Abuse Image Detection
Offensive Image is pervasive in social media. Individuals frequently take advantage of the perceived anonymity of computer-mediated communication, using this to engage in
behavior that many of them would not consider in real life. Online communities, social media platforms, and technology companies have been investing heavily in ways to cope with offensive language in the form of text or images to prevent abusive behavior in social media. The proliferation of, social media platforms resulted in a
remarkable increase in user-generated content. These platforms have empowered users to create, share and exchange content for interacting and communicating with each other. However, these have also opened new avenues to cyberbullies
and haters who can spread their negativity to a larger audience, often anonymously. Due to the pervasiveness and severity of this behavior, many automated approaches that
employ natural language processing (NLP), machine learning and deep learning techniques have been proposed in the past. This survey offers an extensive overview of the stateof-the-art approaches proposed by the research community to identify offensive content. Based on our comprehensive literature survey, a categorization of different approaches and features employed by the researchers in the detection process
are presented. This survey also incorporates the major challenges that require considerable research efforts in this domain. Finally, future research directions with an aim of developing robust abusive content detection system for social
media are also discussed. “The Online Criminal Harms Act will empower the Singapore
government to issue directions to individuals, entities, online service providers and app stores requiring them to remove or block access to potentially criminal content.”

---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name  | Student ID (MTech Applicable)  | Work Items (Who Did What) | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
| Thota Siva Krishna Vara Prasad  | A0249887B | Child Abuse Images Detection , Abuse Language Detection , Community Help Web Portal , FAQChatbot| e0943696@u.nus.edu |


---

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

[![Sudoku AI Solver](https://github.com/sivakrishnathota5/IRS-PM-2023-10-31-IS01PT-CommunityHelpSG-AbuseDetection/blob/main/Miscellaneous/IRS%20Home.png)](https://youtu.be/-AiYLUjP6o8 "Sudoku AI Solver")

---

## About The Project

An open source social network inspired by [reddit](https://www.reddit.com/) built with [Python](https://www.python.org/) using the [Django Web Framework](https://www.djangoproject.com/), trivial templates with Bootstrap & jQuery for UI & UX,  a RESTful API for the web client using [Django Rest Framework](http://www.django-rest-framework.org/). I have designed & developed the [Progressive Web App](https://github.com/thetruefuss/elmer-react) using [React.js](https://reactjs.org/) & [Redux](https://redux.js.org/).



### Built With

- [Python](https://www.python.org/) 3.6.x
- [Django Web Framework](https://www.djangoproject.com/) 2.1.x
- [Django Rest Framework](http://www.django-rest-framework.org/) 3.8.x
- [Twitter Bootstrap](https://getbootstrap.com/docs/4.0/getting-started/introduction/) 4.x
- [jQuery](https://api.jquery.com/) 3.x



## Getting Started



### Prerequisites

- [Python](https://www.python.org/) 3.6.x



### Installation

Create new directory:

```shell
$ mkdir CommunityHelp && cd CommunityHelp
```

Create new virtual environment:

```shell
$ python -m venv venv
```

Activate virtual environment:

```shell
$ source venv/bin/activate  (For Linux)
$ venv/Scripts/activate  (For Windows)
```

Clone this repository:

```shell
$ git clone https://github.com/sivakrishnathota5/CommunityHelp.git src && cd src
```

Install requirements:

```shell
$ pip install -r requirements.txt
```

Copy environment variables:

```shell
$ cp .env.example .env
```

Load static files:

```shell
$ python manage.py collectstatic --noinput
```

Check for any project errors:

```shell
$ python manage.py check
```

Run Django migrations to create database tables:

```shell
$ python manage.py migrate
```

Load initial data for flatpages from fixtures folder:

```shell
$ python manage.py loaddata fixtures/flatpages_data.json
```

Populate the database with dummy data (Optional):

```shell
$ python scripts/populate_database.py
```

Run the development server:

```shell
$ python manage.py runserver
```

Verify the deployment by navigating to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your preferred browser.



## Roadmap

See the [open issues](https://github.com/sivakrishnathota5/IRS-PM-2023-10-31-IS01PT-CommunityHelpSG-AbuseDetection/issues) for a list of proposed features (and known issues).



## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact
<p>
  Send me an email to <a href="mailto:sivakrishnathota5@gmail.com">sivakrishnathota5@gmail.com</a>
  <br />
  Find me online:
  <a href="https://www.linkedin.com/in/sivakrishnathota/">LinkedIn</a> &bull;
</p>

---
## SECTION 6 : PROJECT REPORT / PAPER

`Refer to project report at Github Folder: ProjectReport`



---
## SECTION 7 : MISCELLANEOUS

`Refer to Github Folder: Miscellaneous`

### Model Info.xlsx
* Results of survey
* Insights derived, which were subsequently used in our system

---

### <<<<<<<<<<<<<<<<<<<< End of Template >>>>>>>>>>>>>>>>>>>>

---

**This [Machine Reasoning (MR)](https://www.iss.nus.edu.sg/executive-education/course/detail/machine-reasoning "Machine Reasoning") course is part of the Analytics and Intelligent Systems and Graduate Certificate in [Intelligent Reasoning Systems (IRS)](https://www.iss.nus.edu.sg/stackable-certificate-programmes/intelligent-systems "Intelligent Reasoning Systems") series offered by [NUS-ISS](https://www.iss.nus.edu.sg "Institute of Systems Science, National University of Singapore").**

**Lecturer: [GU Zhan (Sam)](https://www.iss.nus.edu.sg/about-us/staff/detail/201/GU%20Zhan "GU Zhan (Sam)")**

[![alt text](https://www.iss.nus.edu.sg/images/default-source/About-Us/7.6.1-teaching-staff/sam-website.tmb-.png?Status=Master&sfvrsn=5039c05f_2)

**zhan.gu@nus.edu.sg**
