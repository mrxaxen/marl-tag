# MARL Tag implementation in ROS2 and Gazebo

## Dev environment setup (VSCode)

0. Download [VSCode](https://code.visualstudio.com/download) and install with the following command:
`sudo dpkg -i <path-to-deb-file>`
Install [ROS2 Humble Hawksbill](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html), and clone the repository.

1.  Open up the repository in VSCode and go to the extensions tab. Look for `Colcon Tasks` ,`Python(by MS)` and `ROS2(by nonanonno)`

2. Press `Ctrl + Shift + P` and run the following
    * `Colcon: Refresh environment`

3. After the above are finished, you **might** need to edit the file: `.vscode/settings.json` to have the path corrected for ROS2 Humble Hawksbill. The necessary environment variables should be listed in `.vscode/colcon.env` aswell. **They do follow the changes of the environment, however VSCode needs to be restarted, and Colcon: Refresh environment needs to be called**

4. Install [gazebo](https://gazebosim.org/docs/harmonic/install).

5. Install [poetry](https://python-poetry.org/docs/) through the `curl` method, and don't forget to export poetry to path after the installation:
    * `export PATH="~/.local/bin:$PATH"`
    * Check if it was installed properly by opening a new terminal and typing `poetry --version`

6. Make sure you're using `python 3.10`, and have `python3-pip` and `python3.10-venv` installed through `apt`.
    * If not use one or all of the following:
        * `sudo apt install python3.10`
        * `sudo apt install python3-pip`
        * `sudo apt install python3.10-venv`

7. Navigate into the `marl-tag` repository and execute the following:
    * `python3 -m venv .venv --system-site-packages --symlinks`

8. With this you should have a `.venv` folder inside the repository with some folders and files including `bin/activate`. This file activates the virtual environment which we'll use to have workspace specific dependencies, the `--system-site-packages --symlinks` flags are there to ensure accessibility to the ROS underlay. You should only need to use the `activate` script when doing work in the repository from outside VSCode, or if upon opening a terminal from within VSCode, you cannot see the `(.venv) ` prefix in your terminal.  
***From now on everything related to the project must happen with the virtual environment active!***

9. Execute: `poetry install`

10. You should be good to go with the environment install. To run the code within use:
    * `Shift+Ctrl+P`
    * `Tasks: Run task`
    * `python3: dep symlink build`
    * Don't forget to `source install/setup.bash`