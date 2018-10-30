#include <iostream>
#include <string>
#include <vector>

#include "opencv_version.hpp"
#ifdef OPENCV_2
  #include <opencv2/core/gpumat.hpp> // gpu::getCudaEnabledDeviceCount
  #include <opencv2/core/core.hpp>           // Mat, Point, Scalar, Size
  #include <opencv2/contrib/contrib.hpp>
  #include "opencv2/opencv.hpp"
#else
  #include <opencv2/core.hpp>                // Mat, Point, Scalar, Size
  #include <opencv2/core/cuda.hpp>   // cuda::getCudaEnabledDeviceCount
#endif

#include <Python.h>

#include "face_detector.hpp"

using namespace cv;

// #include "pyopencv_generated_const_reg.h"
// #include "pyopencv_generated_types.h"
// #include "pyopencv_generated_type_reg.h"
//#include "pyopencv_generated_funcs.h"
//#include "pyopencv_generated_func_tab.h"

//#include "cv2.cv.hpp"

#if defined(_MSC_VER) && (_MSC_VER >= 1800)
// eliminating duplicated round() declaration
#define HAVE_ROUND 1
#endif



// #if !PYTHON_USE_NUMPY
// #error "The module can only be built if NumPy is available"
// #endif

#define MODULESTR "cv2"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include <numpy/arrayobject.h>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_NONFREE
#  include "opencv2/nonfree/nonfree.hpp"
#endif

using cv::flann::IndexParams;
using cv::flann::SearchParams;

static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

struct ArgInfo
{
    const char * name;
    bool outputarg;
    // more fields may be added if necessary

    ArgInfo(const char * name_, bool outputarg_)
        : name(name_)
        , outputarg(outputarg_) {}

    // to match with older pyopencv_to function signature
    operator const char *() const { return name; }
};

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

using namespace cv;

typedef vector<uchar> vector_uchar;
typedef vector<int> vector_int;
typedef vector<float> vector_float;
typedef vector<double> vector_double;
typedef vector<Point> vector_Point;
typedef vector<Point2f> vector_Point2f;
typedef vector<Vec2f> vector_Vec2f;
typedef vector<Vec3f> vector_Vec3f;
typedef vector<Vec4f> vector_Vec4f;
typedef vector<Vec6f> vector_Vec6f;
typedef vector<Vec4i> vector_Vec4i;
typedef vector<Rect> vector_Rect;
typedef vector<KeyPoint> vector_KeyPoint;
typedef vector<Mat> vector_Mat;
typedef vector<DMatch> vector_DMatch;
typedef vector<string> vector_string;
typedef vector<vector<Point> > vector_vector_Point;
typedef vector<vector<Point2f> > vector_vector_Point2f;
typedef vector<vector<Point3f> > vector_vector_Point3f;
typedef vector<vector<DMatch> > vector_vector_DMatch;

typedef Ptr<Algorithm> Ptr_Algorithm;
typedef Ptr<FeatureDetector> Ptr_FeatureDetector;
typedef Ptr<DescriptorExtractor> Ptr_DescriptorExtractor;
typedef Ptr<Feature2D> Ptr_Feature2D;
typedef Ptr<DescriptorMatcher> Ptr_DescriptorMatcher;
typedef Ptr<CLAHE> Ptr_CLAHE;

typedef SimpleBlobDetector::Params SimpleBlobDetector_Params;

typedef cvflann::flann_distance_t cvflann_flann_distance_t;
typedef cvflann::flann_algorithm_t cvflann_flann_algorithm_t;
typedef Ptr<flann::IndexParams> Ptr_flann_IndexParams;
typedef Ptr<flann::SearchParams> Ptr_flann_SearchParams;

typedef Ptr<FaceRecognizer> Ptr_FaceRecognizer;
typedef vector<Scalar> vector_Scalar;

static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() {}
    ~NumpyAllocator() {}

    void allocate(int dims, const int* sizes, int type, int*& refcount,
                  uchar*& datastart, uchar*& data, size_t* step)
    {
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                      depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                      depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                      depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i;
        npy_intp _sizes[CV_MAX_DIM+1];
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
        {
            /*if( _sizes[dims-1] == 1 )
                _sizes[dims-1] = cn;
            else*/
                _sizes[dims++] = cn;
        }
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
            CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        refcount = refcountFromPyObject(o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( i = 0; i < dims - (cn > 1); i++ )
            step[i] = (size_t)_strides[i];
        datastart = data = (uchar*)PyArray_DATA((PyArrayObject*) o);
    }

    void deallocate(int* refcount, uchar*, uchar*)
    {
        PyEnsureGIL gil;
        if( !refcount )
            return;
        PyObject* o = pyObjectFromRefcount(refcount);
        Py_INCREF(o);
        Py_DECREF(o);
    }
};

NumpyAllocator g_numpyAllocator;

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

// special case, when the convertor needs full ArgInfo structure
static int pyopencv_to(const PyObject* o, Mat& m, const ArgInfo info, bool allowND=true)
{
    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
        return true;
    }

    //std::cout << "debug pt 1" << std::endl;

    if( PyInt_Check(o) )
    {
        double v[] = {
            static_cast<double>(PyInt_AsLong((PyObject*)o)),
            0.,
            0.,
            0.,
        };
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    //std::cout << "debug pt 2" << std::endl;

    if( PyFloat_Check(o) )
    {
        double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    //std::cout << "debug pt 3" << std::endl;

    if( PyTuple_Check(o) )
    {
        int i, sz = (int)PyTuple_Size((PyObject*)o);
        m = Mat(sz, 1, CV_64F);
        for( i = 0; i < sz; i++ )
        {
            PyObject* oi = PyTuple_GET_ITEM(o, i);
            if( PyInt_Check(oi) )
                m.at<double>(i) = (double)PyInt_AsLong(oi);
            else if( PyFloat_Check(oi) )
                m.at<double>(i) = (double)PyFloat_AsDouble(oi);
            else
            {
                failmsg("%s is not a numerical tuple", info.name);
                m.release();
                return false;
            }
        }
        return true;
    }

    //std::cout << "debug pt 4" << std::endl;

    if( !PyArray_Check(o) )
    {
        //std::cout << "debug pt 4.5" << std::endl;
        failmsg("%s is not a numpy array, neither a scalar", info.name);
        return false;
    }

    //std::cout << "debug pt 5" << std::endl;

    PyArrayObject* oarr = (PyArrayObject*) o;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    //std::cout << "debug pt 6" << std::endl;

    if( type < 0 )
    {
        if( typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG )
        {
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = CV_32S;
        }
        else
        {
            failmsg("%s data type = %d is not supported", info.name, typenum);
            return false;
        }
    }

    //std::cout << "debug pt 7" << std::endl;

    int ndims = PyArray_NDIM(oarr);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", info.name, ndims);
        return false;
    }

    //std::cout << "debug pt 8" << std::endl;

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

    //std::cout << "debug pt 9" << std::endl;

    for( int i = ndims-1; i >= 0 && !needcopy; i-- )
    {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        if( (i == ndims-1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims-1 && _strides[i] < _strides[i+1]) )
            needcopy = true;
    }

    //std::cout << "debug pt 10" << std::endl;

    if( ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2] )
        needcopy = true;

    if (needcopy)
    {
        if (info.outputarg)
        {
            failmsg("Layout of the output array %s is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)", info.name);
            return false;
        }

        if( needcast ) {
            o = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*) o;
        }
        else {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            o = (PyObject*) oarr;
        }

        _strides = PyArray_STRIDES(oarr);
    }

    //std::cout << "debug pt 11" << std::endl;

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    //std::cout << "debug pt 12" << std::endl;

    // handle degenerate case
    if( ndims == 0) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    //std::cout << "debug pt 13" << std::endl;

    if( ismultichannel )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    //std::cout << "debug pt 14" << std::endl;

    if( ndims > 2 && !allowND )
    {
        failmsg("%s has more than 2 dimensions", info.name);
        return false;
    }

    //std::cout << "debug pt 15" << std::endl;

    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);

    if( m.data )
    {
        m.refcount = refcountFromPyObject(o);
        if (!needcopy)
        {
            m.addref(); // protect the original numpy array from deallocation
                        // (since Mat destructor will decrement the reference counter)
        }
    };
    //std::cout << "debug pt 16" << std::endl;
    m.allocator = &g_numpyAllocator;

    return true;
}


static FaceDetector detector;
static bool init = false;

//it is only a trick to ensure import_array() is called, when *.so is loaded
//just called only once
void init_numpy(){
     Py_Initialize();
     import_array(); // PyError if not successful
}


bool init_detector(PyObject* args, bool gpu)
{
  init_numpy();
  const char* cascade_file;
  if (init)
  {
    std::cout << "Warning:: Detector already exists" << std::endl;
  }
  else if (PyArg_ParseTuple(args, "s", &cascade_file))
  {
    detector.Init(std::string(cascade_file), gpu);
    init = true;
    std::cout << "Warning:: Detector already exists" << std::endl;
  }
  return false;
}

static PyObject* init_gpu_detector(PyObject* self, PyObject* args)
{
  if (init_detector(args, true))
  {
    Py_RETURN_TRUE;
  }
  else
  {
    Py_RETURN_FALSE;
  }
}

static PyObject* init_cpu_detector(PyObject* self, PyObject* args)
{
  if (init_detector(args, false))
  {
    Py_RETURN_TRUE;
  }
  else
  {
    Py_RETURN_FALSE;
  }
}


//const static int numpy_initialized =  init_numpy();

static PyObject* find_faces(PyObject* self, PyObject* args)
{
  //const char* image_as_chars;
  //PyObject *o;
  if (init)
  {
    //std::cout << "It is init, can find faces" << std::endl;
    //if (PyArg_ParseTuple(args, "s", &image_as_chars))
    //if (PyArg_ParseTuple(args, "O", &o))
    if (true)
    {

      // PyObject *ao = PyObject_GetAttrString(o, "__array_struct__");
      // PyObject *retval;
      
      // if ((ao == NULL) || !PyCObject_Check(ao)) {
      //     std::cout << "object does not have array interface 1" << std::endl;
      //     PyErr_SetString(PyExc_TypeError, "object does not have array interface");
      //     return NULL;
      // }
      
      // PyArrayInterface *pai = (PyArrayInterface*)PyCObject_AsVoidPtr(ao);
      // if (pai->two != 2) {
      //     std::cout << "object does not have array interface 2" << std::endl;
      //     PyErr_SetString(PyExc_TypeError, "object does not have array interface");
      //     Py_DECREF(ao);
      //     return NULL;
      // }
      
      // // Construct data with header info and image data 
      // char *buffer = (char*)pai->data; // The address of image data
      // int width = pai->shape[1];       // image width
      // int height = pai->shape[0];      // image height

      // cv::Mat img(cv::Size(width, height), CV_8UC1, buffer);
      PyObject* pyobj_img = NULL;
      PyArg_ParseTuple(args, "O", &pyobj_img);
      //std::cout << "after PyArg_ParseTuple()" << std::endl;

      cv::Mat img;
      ArgInfo info("abcef", true);
      pyopencv_to(pyobj_img, img, info);
      //std::cout << "after pyopencv_to()" << std::endl;

      //std::cout << "Can parse the image path, continue" << std::endl;
      std::vector<cv::Rect> face_rects;
      detector.Detect(img, face_rects);
      //std::cout << "can call detector.Detect()" << std::endl;
      if (face_rects.empty())
      {
        // No faces
        return PyList_New(0);
      }
      PyObject* face_list = PyList_New(face_rects.size());
      for (int i = 0; i < (int)face_rects.size(); ++i)
      {
        cv::Point point = face_rects[i].tl();
        cv::Size dims = face_rects[i].size();
        PyObject* face_rect = PyTuple_New(4);
        PyTuple_SetItem(face_rect, 0, Py_BuildValue("i", point.x));
        PyTuple_SetItem(face_rect, 1, Py_BuildValue("i", point.y));
        PyTuple_SetItem(face_rect, 2, Py_BuildValue("i", dims.width));
        PyTuple_SetItem(face_rect, 3, Py_BuildValue("i", dims.height));
        PyList_SetItem(face_list, i, face_rect);
      }
      return face_list;
    }
    std::cout << "Error: Problem parsing image path" << std::endl;
    return PyList_New(0);
  }
  std::cout << "Error: Must call cv2gpu.create_face_recognizer!" << std::endl;
  return PyList_New(0);
}

static PyObject* is_cuda_compatible(PyObject* self, PyObject* args)
{
  #ifdef OPENCV_2
    if (cv::gpu::getCudaEnabledDeviceCount())
    {
      Py_RETURN_TRUE;
    }
  #else
    if (cv::cuda::getCudaEnabledDeviceCount())
    {
      Py_RETURN_TRUE;
    }
  #endif
  Py_RETURN_FALSE;
}

static PyMethodDef cv2gpuMethods[] =
{
  {"init_cpu_detector" , init_cpu_detector , METH_VARARGS, "Initializes CPU OpenCV FaceRecognizer object" },
  {"init_gpu_detector" , init_gpu_detector , METH_VARARGS, "Initializes GPU OpenCV FaceRecognizer object" },
  {"is_cuda_compatible", is_cuda_compatible, METH_NOARGS , "Checks for CUDA compatibility"                },
  {"find_faces"        , find_faces        , METH_VARARGS, "Finds faces using initialized FaceRecognizer" },
  {NULL                , NULL              , 0           , NULL                                           }
};

#if PY_VERSION_HEX >= 0x03000000

/* Python 3.x code */

static struct PyModuleDef cv2gpu =
{
  PyModuleDef_HEAD_INIT,
  "cv2gpu", /* name of module */
  NULL,     /* module documentation, may be NULL */
  -1,       /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
  cv2gpuMethods
};

PyMODINIT_FUNC
PyInit_cv2gpu(void)
{
  (void) PyModule_Create(&cv2gpu);
}

#else

/* Python 2.x code */

PyMODINIT_FUNC
initcv2gpu(void)
{
  (void) Py_InitModule("cv2gpu", cv2gpuMethods);
}

#endif
