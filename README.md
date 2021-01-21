# Anomaly detection experimentation app
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


## About The Project
![image](anomalydetection_app/assets/demo.png)

This project defines a Dash app to explore anomaly detection for time series data.

The app allows you to 
* select which data patterns and types of noise to simulate
* try different anomaly detection approaches to see their effect on the simulated data. 

### Built With

The following projects were used to build this project:
* [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)
* [Dash](https://plotly.com/dash/)
* [Plotly](https://plotly.com/)
* [anomalydetection](https://github.com/DHI/anomalydetection/)


## Getting Started

This section contains instructions to run the app locally.

### Download code
First, make sure you have the code locally. There are different ways to download the code:

* Clone using git if you have git installed: 
   ```sh
   git clone https://github.com/DHI/anomalydetection_app.git
   ```
 * Or download a zip file using the "Code" button on https://github.com/DHI/anomalydetection_app

### Installation
You can either install this project as a package using Poetry, or install the requirements.

Installing in a virtual environment is recommended and avoids conflicts between dependencies
in different projects. To create and activate a virtual environment, do the following:

1. Create a virtual environment named 'env': \
`python -m venv env`

2. Activate the virtual environment: \
`env\Scripts\activate`

If you choose not to install in a virtual environment, you can skip these steps.

#### Install project as package
This project uses Poetry for package management (see https://github.com/sdispater/poetry for more information). 
To install using Poetry, follow these steps:

1. Install Poetry: \
`curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python` \

2. In the command prompt, navigate to the root of this project (the folder where this README.md file is located)

3. Install the packages with specified versions defined in poetry.lock from the command prompt with: \
`poetry install`

#### Install project requirements
Install the requirements defined in requirements.txt with:
`pip install -r requirements.txt`


### Running app
To run this app, do the following:

1. In the command prompt, navigate to the folder where app.py is located.
2. Run the following from the command line:
`python app.py`
3. The app will open in a browser.


## Roadmap

See the [open issues](https://github.com/DHI/anomalydetection_app/issues) for a list of proposed features 
(and known issues).


## Contributing
You are very welcome to open pull requests with new features, e.g. new noise types or data patterns to simulate,
more buttons to control input to anomaly detectors and so on.
Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact
Project Link: [https://github.com/DHI/anomalydetection_app](https://github.com/DHI/anomalydetection_app)


## Acknowledgements
* [Best README template](https://github.com/othneildrew/Best-README-Template/blob/master/README.md)
* [Choose an Open Source License](https://choosealicense.com)
