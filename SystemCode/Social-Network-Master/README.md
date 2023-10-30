
<br />

<p align="center">
  <a href="https://github.com/thetruefuss/elmer">
    <img src="https://github.com/sivakrishnathota5/CommunityHelp/blob/main/Social-Network-Master/static/img/logos.png" alt="Logo" width="50" height="50">
  </a>

  <h3 align="center">Singapore Community Help</h3>

  <p align="center">
    <br />
    <a href="http://51.79.71.46:8000/"><strong>Explore the app »</strong></a>
    <br />
    <br />
  </p>
</p>



<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#screenshots">Screenshots</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



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

See the [open issues](https://github.com/sivakrishnathota5/CommunityHelp/issues) for a list of proposed features (and known issues).



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




[linkedin-url]: https://www.linkedin.com/in/sivakrishnathota/
