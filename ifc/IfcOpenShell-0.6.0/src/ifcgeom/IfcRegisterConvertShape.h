#include "IfcRegisterUndef.h"
#define SHAPE(T) \
	if ( !processed && l->declaration().is(IfcSchema::T::Class()) ) { \
		processed = true; \
		try { \
			if ( convert((IfcSchema::T*)l,r) ) { \
				success = true; \
			} \
		} catch (const std::exception& e) { \
			Logger::Message(Logger::LOG_ERROR, std::string(e.what()) + "\nFailed to convert:", l); \
			return false; \
		} catch (const Standard_Failure& f) { \
			if (f.GetMessageString() && strlen(f.GetMessageString())) \
				Logger::Message(Logger::LOG_ERROR, std::string("Error in: ") + f.GetMessageString() + "\nFailed to convert:", l); \
			else \
				Logger::Message(Logger::LOG_ERROR, "Failed to convert:", l); \
			return false; \
		} \
		if (!success) { \
			Logger::Message(Logger::LOG_ERROR,"Failed to convert:",l); \
			return false; \
		} \
	}
#include "IfcRegisterDef.h"

#include "IfcRegister.h"