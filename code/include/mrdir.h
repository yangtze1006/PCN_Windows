#ifndef MRDIR_H
#define MRDIR_H
#include "string"
#include "iostream"
#include "vector"

//////////////////////////////////////////////////////////////////////////
///	����ʵ�ֿ�ƽ̨�ļ�����
///	EXISTS �ж��ļ��Ƿ����
///	MKDIR  �½��ļ���(�����𼶴���)
///	SLEEP  ���ߺ���
//////////////////////////////////////////////////////////////////////////
#ifdef _WIN32
#include <io.h>
#include <direct.h>
#include <windows.h>
#define EXISTS(path) (_access(path, 0)!=-1)
#define MKDIR(path) _mkdir(path)
#define SLEEP(ms) Sleep(ms)
static bool exist(const char*filepath){return (_access(filepath, 0)) != -1;}
#else
#include <unistd.h>
#include <sys/stat.h>
#define EXISTS(path) (access(path, 0)!=-1)
#define MKDIR(path) mkdir(path, 0775)
#define SLEEP(ms) usleep(ms)
#endif

//////////////////////////////////////////////////////////////////////////
///	 @brief����ʵ��Windows���ַ���ת������ز���
///	ANSIToUnicode 
///	UnicodeToANSI
///	UTF8ToUnicode
///	UnicodeToUTF8
///	ANSIToUTF8
///	UTF8ToANSI
///	@param str ��ת���ַ�
///	@retval ת������ַ�
//////////////////////////////////////////////////////////////////////////

#ifdef _WIN32
#include <tchar.h>
#include "windows.h"
#if _UNICODE
static wchar_t * ANSIToUnicode(const char* str)
{
	if (!str)
		return NULL;
	int textlen;
	wchar_t * result;
	textlen = MultiByteToWideChar(CP_ACP, 0, str, -1, NULL, 0);
	result = (wchar_t *)malloc((textlen + 1)*sizeof(wchar_t));
	memset(result, 0, (textlen + 1)*sizeof(wchar_t));
	MultiByteToWideChar(CP_ACP, 0, str, -1, (LPWSTR)result, textlen);
	result[textlen] = '\0';
	return result;
}
static char * UnicodeToANSI(const wchar_t* str)
{
	if (!str)
		return NULL;
	char* result;
	int textlen;
	textlen = WideCharToMultiByte(CP_ACP, 0, str, -1, NULL, 0, NULL, NULL);
	result = (char *)malloc((textlen + 1)*sizeof(char));
	memset(result, 0, sizeof(char)* (textlen + 1));
	WideCharToMultiByte(CP_ACP, 0, str, -1, result, textlen, NULL, NULL);
	result[textlen] = '\0';
	return result;
}
static wchar_t * UTF8ToUnicode(const char* str)
{
	int textlen;
	wchar_t * result;
	textlen = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
	result = (wchar_t *)malloc((textlen + 1)*sizeof(wchar_t));
	memset(result, 0, (textlen + 1)*sizeof(wchar_t));
	MultiByteToWideChar(CP_UTF8, 0, str, -1, (LPWSTR)result, textlen);
	return result;
}
static char * UnicodeToUTF8(const wchar_t* str)
{
	char* result;
	int textlen;
	textlen = WideCharToMultiByte(CP_UTF8, 0, str, -1, NULL, 0, NULL, NULL);
	result = (char *)malloc((textlen + 1)*sizeof(char));
	memset(result, 0, sizeof(char)* (textlen + 1));
	WideCharToMultiByte(CP_UTF8, 0, str, -1, result, textlen, NULL, NULL);
	return result;
}
static char* ANSIToUTF8(const char* str)
{
	return UnicodeToUTF8(ANSIToUnicode(str));
}
static char* UTF8ToANSI(const char* str)
{
	return UnicodeToANSI(UTF8ToUnicode(str));
}
#endif

//////////////////////////////////////////////////////////////////////////
///  @brief ��ȡ��ǰ�ļ������������ļ���
///  @param strDir Ҫ��ȡ���ļ���·��
///  @param subdirs ��ȡ�������е���Ŀ¼����
//////////////////////////////////////////////////////////////////////////
static std::vector<std::string> getAllSubdirs(std::string strDir)
{
	WIN32_FIND_DATA FindData;
	HANDLE hError;
	std::string file2find = strDir + "/*.*";
	std::vector<std::string> subdirs;
#if _UNICODE
	wchar_t *p = ANSIToUnicode(file2find.c_str());
	hError = FindFirstFile(p, &FindData);
	delete[]p;
#else
	hError = FindFirstFile((LPCTSTR)file2find.c_str(), &FindData);
#endif
	if (hError == INVALID_HANDLE_VALUE)
	{
		std::cout << "cannot find subdir in " << strDir<< std::endl;
		return subdirs;
	}
	else
	{
		do
		{
			if (lstrcmp(FindData.cFileName, TEXT(".")) == 0 || lstrcmp(FindData.cFileName, TEXT("..")) == 0)
			{
				continue;
			}
			if (!(FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				continue;
			}
			if (FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
#if _UNICODE
				char*p = UnicodeToANSI(FindData.cFileName);
				subdirs.push_back(p);
				delete[]p;
#else
				subdirs.push_back(FindData.cFileName);
#endif
			}
		} while (::FindNextFile(hError, &FindData));
	}
	FindClose(hError);
	return subdirs;
}
//////////////////////////////////////////////////////////////////////////
///  @brief ��ȡ��ǰ�ļ����������ļ�
///  @param strDir Ҫ��ȡ���ļ���·��
///  @param subdirs ��ȡ�������е��ļ�����
///  @param ext ��׺��
//////////////////////////////////////////////////////////////////////////
static std::vector<std::string> getAllFilesinDir(std::string strDir, std::string ext = "*.*")
{
	WIN32_FIND_DATA FindData;
	HANDLE hError;
	std::string file2find = strDir + "/" + ext;
	std::vector<std::string> files;
#if _UNICODE
	wchar_t *p= ANSIToUnicode(file2find.c_str());
	hError = FindFirstFile(p, &FindData);
	delete[]p;
#else
	hError = FindFirstFile((LPCTSTR)file2find.c_str(), &FindData);
#endif
	if (hError == INVALID_HANDLE_VALUE)
	{
		return files;
	}
	else
	{
		do
		{
			if (lstrcmp(FindData.cFileName, TEXT(".")) == 0 || lstrcmp(FindData.cFileName, TEXT("..")) == 0)
			{
				continue;
			}
			else if (FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				continue;
			}
			else if (!(FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
#if _UNICODE
				char *p = UnicodeToANSI(FindData.cFileName);
				files.push_back(p);
				delete[]p;
#else
				files.push_back(FindData.cFileName);
#endif
			}
		} while (::FindNextFile(hError, &FindData));
	}
	FindClose(hError);
	return files;
}

#else
#include <dirent.h>  
#include <sys/types.h>  
#include <sys/stat.h>  
#include <unistd.h>
#include <string.h>  
using std::string;
static std::vector<std::string> getAllSubdirs(std::string strDir)
{
	DIR *dp;
	struct dirent *entry;
	std::vector<std::string> subdirs;
	if((dp = opendir(strDir.c_str())) == NULL)
	{
		closedir(dp);
		return subdirs;
	}
	while((entry = readdir(dp)) != NULL)
	{
		if (strcmp(".", entry->d_name) == 0 || strcmp("..", entry->d_name) == 0)
			continue;
		if(entry->d_type==4)
		{
			subdirs.push_back(entry->d_name); 
		} 
	} 
	closedir(dp);
	return subdirs;
}
static std::vector<std::string>getAllFilesinDir(std::string strDir, std::string ext="*.*")
{
	struct dirent* ent = NULL;  
	DIR *pDir;
	std::vector<std::string> files;
	pDir = opendir(strDir.c_str());  
	if (pDir == NULL) {
		closedir(pDir);
		return files;
	}
	while (NULL != (ent = readdir(pDir))) {  
		if (ent->d_type == 8) {  
			files.push_back(ent->d_name);
		}
		else {
			if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
				continue;
			}
		}
	}
	closedir(pDir);  
	return files;
}
#endif
#endif
