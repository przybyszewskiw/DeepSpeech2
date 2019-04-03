#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#include <Python.h>
#include <vector>
#include <string>
#include "ctcbeam.hpp"

static PyObject *
ctcBeamWrapper(PyObject *dummy, PyObject *args)
{
    PyObject *lst=NULL;
    const char *file;

    if (!PyArg_ParseTuple(args, "Os", &lst, &file)) return NULL;

    if (!PyList_Check(lst)) return NULL;
    int len = PyList_Size(lst);
    std::vector<std::vector<double>> vec;
    
    for (int i = 0; i < len; i++)
    {
    	vec.push_back(std::vector<double>{});
    	PyObject* row = PyList_GetItem(lst, i);
    	int rowL = PyList_Size(row);
        for (int j = 0; j < rowL; j++)
        {
            PyObject *elem = PyList_GetItem(row, j);
            vec[i].push_back(PyFloat_AS_DOUBLE(elem));
        }
    }

    std::string result = ctcBeamSearch(vec, 100, globalClasses, languageModelFromFile(file));
    return PyUnicode_FromString(result.c_str());
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
    return PyModule_Create(&ctcBeamModule);
}


int main(int argc, char **argv)
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL)
    {
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
