#ifndef MR_DUMPER_H_
#define MR_DUMPER_H_

#if _WIN32
	#include <windows.h>
	#include <string>
	#include <dbghelp.h>
	#pragma comment(lib, "dbghelp.lib")

namespace mrdumper
{
	#ifdef UNICODE    
	#define tstring wstring     
	#else       
	#define tstring string 
	#endif

	class MRDumper
	{
	public:
		MRDumper();
		~MRDumper();
		static bool _PlaceHolder();
	private:
		LPTOP_LEVEL_EXCEPTION_FILTER m_OriginalFilter;
		static LONG WINAPI ExceptionFilter(struct _EXCEPTION_POINTERS* ExceptionInfo);
	};
	MRDumper::MRDumper()
	{
		m_OriginalFilter = SetUnhandledExceptionFilter(ExceptionFilter);
	}

	MRDumper::~MRDumper()
	{
		SetUnhandledExceptionFilter(m_OriginalFilter);
	}

	LONG WINAPI MRDumper::ExceptionFilter(struct _EXCEPTION_POINTERS* ExceptionInfo)
	{
		bool bDumpOK = false;
		DWORD dwProcess = GetCurrentProcessId();
		HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, dwProcess);
		if (hProcess != INVALID_HANDLE_VALUE)
		{
			TCHAR szPath[MAX_PATH];
			if (GetModuleFileName(NULL, szPath, sizeof(szPath)))
			{
				std::tstring strDumpFileName = szPath;
				strDumpFileName += TEXT(".dmp");
				HANDLE hFile = CreateFile(strDumpFileName.c_str(), FILE_ALL_ACCESS, 0, NULL, CREATE_ALWAYS, NULL, NULL);
				if (hFile != INVALID_HANDLE_VALUE)
				{
					MINIDUMP_EXCEPTION_INFORMATION exception_information;
					exception_information.ThreadId = GetCurrentThreadId();
					exception_information.ExceptionPointers = ExceptionInfo;
					exception_information.ClientPointers = TRUE;
					if (MiniDumpWriteDump(hProcess, dwProcess, hFile, MiniDumpNormal, &exception_information, NULL, NULL))
					{
						bDumpOK = true;
					}
					CloseHandle(hFile);
				}
			}
			CloseHandle(hProcess);
		}

		if (bDumpOK)
			MessageBox(NULL, TEXT("����������δ������쳣��MiniDump�ļ��Ѿ������ڳ��������Ŀ¼��"), TEXT("��ʾ"), MB_OK);
		else
			MessageBox(NULL, TEXT("����������δ������쳣������MiniDump�ļ�ʧ�ܡ�"), TEXT("��ʾ"), MB_OK);
		return EXCEPTION_EXECUTE_HANDLER;
	}
	bool MRDumper::_PlaceHolder() { return true; }
	static MRDumper dumper;
	const bool bPlaceHolder = MRDumper::_PlaceHolder();
}
#endif

#endif