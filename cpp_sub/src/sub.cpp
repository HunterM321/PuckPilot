#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class VideoFrameSubscriber : public rclcpp::Node
{
public:
    VideoFrameSubscriber()
    : Node("video_frame_cpp")
    {
        // Subscription to receive frames from "video_frames" topic
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "video_frames", 10,
            std::bind(&VideoFrameSubscriber::image_callback, this, std::placeholders::_1));
        
        // Publisher to send frames with optical flow visualization to "optical_flow" topic
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("optical_flow", 10);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Convert ROS image message to OpenCV Mat
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;

        // Convert to grayscale for optical flow calculation
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (!prev_gray_.empty())  // Ensure we have a previous frame
        {
            // Calculate optical flow between previous and current frame
            cv::Mat flow;
            cv::calcOpticalFlowFarneback(prev_gray_, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

            // Draw the optical flow using a color wheel representation
            cv::Mat color_wheel_frame;
            draw_optical_flow_color_wheel(flow, color_wheel_frame);

            // Convert the color wheel frame back to a ROS message
            sensor_msgs::msg::Image::SharedPtr output_msg = cv_bridge::CvImage(msg->header, "bgr8", color_wheel_frame).toImageMsg();

            // Publish the annotated frame
            publisher_->publish(*output_msg);
        }

        // Update the previous frame
        prev_gray_ = gray.clone();
    }

    void draw_optical_flow_color_wheel(const cv::Mat& flow, cv::Mat& color_wheel_frame)
    {
        // Split the flow into separate x and y components
        std::vector<cv::Mat> flow_channels(2);
        cv::split(flow, flow_channels);
        cv::Mat flow_x = flow_channels[0];  // Horizontal flow (x-direction)
        cv::Mat flow_y = flow_channels[1];  // Vertical flow (y-direction)

        // Calculate the magnitude and angle of the flow
        cv::Mat magnitude, angle;
        cv::cartToPolar(flow_x, flow_y, magnitude, angle, true);

        // Normalize the magnitude to fit within the range [0, 1]
        cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

        // Scale angle to [0, 180] for OpenCV HSV
        angle *= 180 / CV_PI; 

        // Create an HSV image where:
        // - Hue corresponds to the angle (direction of motion)
        // - Saturation is set to 1 (fully saturated)
        // - Value corresponds to the normalized magnitude (strength of motion)
        cv::Mat hsv_channels[3];
        hsv_channels[0] = angle;                                 // Hue (direction)
        hsv_channels[1] = cv::Mat::ones(angle.size(), CV_32F) * 255;   // Saturation (full)
        hsv_channels[2] = magnitude * 255;                       // Value (magnitude)

        // Merge HSV channels and convert to BGR for visualization
        cv::Mat hsv;
        cv::merge(hsv_channels, 3, hsv);
        hsv.convertTo(hsv, CV_8U);  // Convert to 8-bit format
        cv::cvtColor(hsv, color_wheel_frame, cv::COLOR_HSV2BGR);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    cv::Mat prev_gray_;  // Previous grayscale frame for optical flow calculation
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoFrameSubscriber>());
    rclcpp::shutdown();
    return 0;
}
