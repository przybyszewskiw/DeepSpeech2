#include <Python.h>
#include "numpy/arrayobject.h"
#include <vector>
#include <string>

/*std::string ctcBeam(std::vector<std::vector<double>> mat, std::string classes )
{
    return "todo"; //TODO
}*/

static PyObject *
ctcBeamWrapper(PyObject *dummy, PyObject *args)
{
    PyObject *arg1=NULL;
    PyObject *arr1=NULL;

    if (!PyArg_ParseTuple(args, "O", &arg1)) return NULL;

    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL) return NULL;

    printf("c++ here:\n");
    int nd = PyArray_NDIM(arr1);
    npy_intp *lol = PyArray_DIMS(arr1);
    double *dptr = (double *)PyArray_DATA(arr1);
    printf("ndim: %d dims: %d %d\n", nd, (int)lol[0], (int)lol[1]);
    for (int i = 0; i < lol[0]; i++)
    {
        for (int j = 0; j < lol[1]; j++)
        {
            printf("%lf ", dptr[i * lol[1] + j]);
        }
        printf("\n");
    }
    //here we count ctcbeam

    Py_DECREF(arr1);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef ctcBeamMethods[] =
{
    {"ctcbeam", ctcBeamWrapper, METH_VARARGS, "count ctc transcript through beam search"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ctcBeamModule =
{
    PyModuleDef_HEAD_INIT,
    "ctcbeam",
    NULL,
    -1,
    ctcBeamMethods
};

PyMODINIT_FUNC PyInit_ctcbeam(void)
{
    import_array();
    return PyModule_Create(&ctcBeamModule);
}


int main(int argc, char **argv)
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    PyImport_AppendInittab("ctcbeam", PyInit_ctcbeam);

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    PyMem_RawFree(program);
    return 0;
}
