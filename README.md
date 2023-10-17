# MARL Tag implementation in ROS2 and Gazebo

## Dev environment setup (VSCode)

0. Download [VSCode](https://code.visualstudio.com/download) and install with the following command:
`sudo dpkg -i <path-to-deb-file>`
Install [ROS2 Humble Hawksbill](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html), and clone the repository.

1.  Open up the repository in VSCode and go to the extensions tab. Look for `Colcon Tasks` ,`Python(by MS)` and `ROS2(by nonanonno)`

2. Press `Ctrl + Shift + P` and run the following
    * `Colcon: Enable tasks for current workspace`
    * `Colcon: Refresh environment`

3. After the above are finished, you'll need to edit the file: `.vscode/settings.json` to have the path corrected for ROS2 Humble Hawksbill. The necessary environment variables should be listed in `.vscode/colcon.env` aswell. **They do follow the changes of the environment, however VSCode needs to be restarted, and Colcon: Refresh environment needs to be called**

4. Install [gazebo](https://gazebosim.org/docs/harmonic/install).