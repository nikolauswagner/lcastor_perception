cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(lcastor_perception)

find_package(catkin REQUIRED COMPONENTS
	roslaunch
	rospy
	message_generation
	vision_msgs
)

catkin_python_setup()

add_message_files(FILES

)

add_service_files(FILES
  DetectObjects.srv
)

generate_messages(DEPENDENCIES
	vision_msgs
)

catkin_package(
CATKIN_DEPENDS 
	roslaunch
	rospy
)
