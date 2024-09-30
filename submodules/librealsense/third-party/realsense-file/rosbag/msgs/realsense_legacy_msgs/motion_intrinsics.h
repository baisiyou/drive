// Generated by gencpp from file realsense_legacy_msgs/motion_intrinsics.msg
// DO NOT EDIT!


#ifndef realsense_legacy_msgs_MESSAGE_MOTION_INTRINSICS_H
#define realsense_legacy_msgs_MESSAGE_MOTION_INTRINSICS_H


#include <string>
#include <vector>
#include <array>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace realsense_legacy_msgs
{
template <class ContainerAllocator>
struct motion_intrinsics_
{
  typedef motion_intrinsics_<ContainerAllocator> Type;

  motion_intrinsics_()
    : data()
    , noise_variances()
    , bias_variances()  {
      data.fill(0.0);

      noise_variances.fill(0.0);

      bias_variances.fill(0.0);
  }
  motion_intrinsics_(const ContainerAllocator& _alloc)
    : data()
    , noise_variances()
    , bias_variances()  {
  (void)_alloc;
      data.fill(0.0);

      noise_variances.fill(0.0);

      bias_variances.fill(0.0);
  }



   typedef std::array<float, 12>  _data_type;
  _data_type data;

   typedef std::array<float, 3>  _noise_variances_type;
  _noise_variances_type noise_variances;

   typedef std::array<float, 3>  _bias_variances_type;
  _bias_variances_type bias_variances;




  typedef std::shared_ptr< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> > Ptr;
  typedef std::shared_ptr< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> const> ConstPtr;

}; // struct motion_intrinsics_

typedef ::realsense_legacy_msgs::motion_intrinsics_<std::allocator<void> > motion_intrinsics;

typedef std::shared_ptr< ::realsense_legacy_msgs::motion_intrinsics > motion_intrinsicsPtr;
typedef std::shared_ptr< ::realsense_legacy_msgs::motion_intrinsics const> motion_intrinsicsConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> & v)
{
rs2rosinternal::message_operations::Printer< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace realsense_legacy_msgs

namespace rs2rosinternal
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'realsense_legacy_msgs': ['/home/administrator/realsense_ros_file/realsense_file/realsense_legacy_msgs/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> >
  : std::true_type
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> const>
  : std::true_type
  { };

template <class ContainerAllocator>
struct IsMessage< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> >
  : std::true_type
  { };

template <class ContainerAllocator>
struct IsMessage< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> const>
  : std::true_type
  { };

template <class ContainerAllocator>
struct HasHeader< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> >
  : std::false_type
  { };

template <class ContainerAllocator>
struct HasHeader< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> const>
  : std::false_type
  { };


template<class ContainerAllocator>
struct MD5Sum< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> >
{
  static const char* value()
  {
    return "aebdc2f8f9726f1c3ca823ab56e47429";
  }

  static const char* value(const ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xaebdc2f8f9726f1cULL;
  static const uint64_t static_value2 = 0x3ca823ab56e47429ULL;
};

template<class ContainerAllocator>
struct DataType< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> >
{
  static const char* value()
  {
    return "realsense_legacy_msgs/motion_intrinsics";
  }

  static const char* value(const ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32[12] data\n\
float32[3] noise_variances\n\
float32[3] bias_variances\n\
";
  }

  static const char* value(const ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace rs2rosinternal

namespace rs2rosinternal
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.data);
      stream.next(m.noise_variances);
      stream.next(m.bias_variances);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct motion_intrinsics_

} // namespace serialization
} // namespace rs2rosinternal

namespace rs2rosinternal
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::realsense_legacy_msgs::motion_intrinsics_<ContainerAllocator>& v)
  {
    s << indent << "data[]" << std::endl;
    for (size_t i = 0; i < v.data.size(); ++i)
    {
      s << indent << "  data[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.data[i]);
    }
    s << indent << "noise_variances[]" << std::endl;
    for (size_t i = 0; i < v.noise_variances.size(); ++i)
    {
      s << indent << "  noise_variances[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.noise_variances[i]);
    }
    s << indent << "bias_variances[]" << std::endl;
    for (size_t i = 0; i < v.bias_variances.size(); ++i)
    {
      s << indent << "  bias_variances[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.bias_variances[i]);
    }
  }
};

} // namespace message_operations
} // namespace rs2rosinternal

#endif // realsense_legacy_msgs_MESSAGE_MOTION_INTRINSICS_H
